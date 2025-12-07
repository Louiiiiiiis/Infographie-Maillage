#include <igl/readOFF.h>
#include <igl/adjacency_list.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>
#include <igl/min_quad_with_fixed.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <queue>
#include <set>
#include <iostream>

// ==========================================
// CONSTANTES & STRUCTURES
// ==========================================

const unsigned int KEY_PLUS = 334;  // Agrandir pinceau
const unsigned int KEY_MINUS = 333; // Rétrécir pinceau
const unsigned int KEY_SPACE = 32;  // Calculer Poids
const unsigned int KEY_J = 74;      // Changer Fonction Transfert
const unsigned int KEY_C = 67;      // Clear Poignée
const unsigned int KEY_R = 82;      // Reset
// Flèches directionnelles
const unsigned int KEY_UP = 265;
const unsigned int KEY_DOWN = 264;
const unsigned int KEY_LEFT = 263;
const unsigned int KEY_RIGHT = 262;

struct MeshApp {
    Eigen::MatrixXd V, V_original, C; // Sommets, Backup, Couleurs
    Eigen::MatrixXi F;                // Faces
    std::vector<std::vector<int>> A;  // Listes d'adjacence
    
    // Sélection
    int k = 2; // Taille du pinceau (k-anneaux)
    int current_vertex = -1;
    std::set<int> fixed_vertices;  // Zone Bleue (Ancre)
    std::set<int> handle_vertices; // Zone Rouge (Poignée)
    
    // Déformation
    Eigen::VectorXd weights;       // Poids "d'influence" du déplacement (0..1)
    Eigen::Vector3f translation = Eigen::Vector3f::Zero(); // Vecteur déplacement
    bool weights_computed = false;
    int transfer_function = 0;     // 0=Linear, 1=Smooth, 2=Squared
};

// ==========================================
// 1. UTILITAIRES (Sélection)
// ==========================================

// Récupère les voisins à 'k' anneaux autour d'un sommet
std::set<int> get_neighbors(const MeshApp& app, int seed_vertex) {
    std::set<int> visited;
    std::queue<std::pair<int, int>> q;
    q.push({seed_vertex, 0});
    visited.insert(seed_vertex);

    while(!q.empty()) {
        auto [v, depth] = q.front(); q.pop();
        if (depth >= app.k) continue;
        for (int n : app.A[v]) {
            if (!visited.count(n)) {
                visited.insert(n);
                q.push({n, depth + 1});
            }
        }
    }
    return visited;
}

// Initialise la zone fixe (le bas du maillage) automatiquement
void init_fixed_zone(MeshApp& app) {
    double min_y = app.V.col(1).minCoeff();
    double max_y = app.V.col(1).maxCoeff();
    double epsilon = 0.05 * (max_y - min_y); // 5% de la hauteur total pour servir de base

    app.fixed_vertices.clear();
    for(int i=0; i<app.V.rows(); ++i) {
        if(app.V(i, 1) < min_y + epsilon) {
            app.fixed_vertices.insert(i);
        }
    }
    std::cout << "[INFO] Zone Fixe auto-selectionnee : " << app.fixed_vertices.size() << " sommets (Bas).\n";
}

// ==========================================
// 2. VISUALISATION
// ==========================================

void update_colors(MeshApp& app, igl::opengl::glfw::Viewer& viewer) {
    // 1. Si les poids sont calculés, on affiche le dégradé (Heatmap)
    if(app.weights_computed && app.weights.size() == app.V.rows()) {
        igl::jet(app.weights, true, app.C);
    } 
    // 2. Sinon, on affiche les zones de sélection (Bleu/Rouge)
    else {

        if (app.C.rows() != app.V.rows() || app.C.cols() != 3) {
            app.C.resize(app.V.rows(), 3);
        }

        // D'abord, on calcule quels points sont sous la souris (le pinceau vert)
        // pour ne pas le recalculer 10 000 fois dans la boucle.
        std::set<int> points_sous_souris;
        if (app.current_vertex != -1) {
            points_sous_souris = get_neighbors(app, app.current_vertex);
        }

        // MAINTENANT, on visite chaque sommet un par un pour décider sa couleur finale
        for (int i = 0; i < app.V.rows(); i++) {
            
            if (app.handle_vertices.count(i) > 0) {
                app.C(i, 0) = 1.0; 
                app.C(i, 1) = 0.0; 
                app.C(i, 2) = 0.0; // Rouge
            }
            else if (app.fixed_vertices.count(i) > 0) {
                app.C(i, 0) = 0.0; 
                app.C(i, 1) = 0.0; 
                app.C(i, 2) = 1.0; // Bleu
            }
            else if (points_sous_souris.count(i) > 0) {
                app.C(i, 0) = 0.0; 
                app.C(i, 1) = 1.0; 
                app.C(i, 2) = 0.0; // Vert
            }
            else {
                app.C(i, 0) = 0.8; 
                app.C(i, 1) = 0.8; 
                app.C(i, 2) = 0.8; // Gris
            }
        }
    }
    viewer.data().set_colors(app.C);
}

// ==========================================
// 3. CALCUL (Laplacien)
// ==========================================

void compute_harmonic_weights(MeshApp& app) {
    if(app.fixed_vertices.empty() || app.handle_vertices.empty()) {
        std::cout << "[ERREUR] Il manque des points bleus ou rouges !\n";
        return;
    }

    std::cout << "[CALCUL] Lissage iteratif... ";

    app.weights = Eigen::VectorXd::Constant(app.V.rows(), 0.0);

    for (int h : app.handle_vertices) app.weights(h) = 1.0; 

    int iterations = 1000; 
    
    for(int k=0; k<iterations; k++) {
        
        for(int i=0; i<app.V.rows(); i++) {
            
            if(app.fixed_vertices.count(i) || app.handle_vertices.count(i)) {
                continue; 
            }

            double somme_voisins = 0.0;
            double nombre_voisins = 0.0;

            for(int voisin : app.A[i]) {
                somme_voisins += app.weights(voisin);
                nombre_voisins += 1.0;
            }

            if(nombre_voisins > 0) {
                app.weights(i) = somme_voisins / nombre_voisins;
            }
        }
    }

    app.weights_computed = true;
    std::cout << "OK (Diffusé " << iterations << " fois) !\n";
}

// ==========================================
// 4. DEFORMATION
// ==========================================

void apply_deformation(MeshApp& app) {
    if(!app.weights_computed) return;
    
    for(int i=0; i<app.V.rows(); ++i) {
        double w = app.weights(i); // Poids brut (0..1)
        double f_w = w;
        
        // Fonction de transfert
        if(app.transfer_function == 1)      
            f_w = w * w * (3 - 2 * w); // Smoothstep
        else if (app.transfer_function == 2) 
            f_w = w * w;               // Squared
        
        // V_new = V_old + poids * translation
        app.V.row(i) = app.V_original.row(i) + f_w * app.translation.cast<double>().transpose();
    }
}

// ==========================================
// MAIN
// ==========================================

int main(int argc, char *argv[]){
  MeshApp app;
  
  // 1. Chargement
  igl::readOFF("../meshes/bunny.off", app.V, app.F);
  app.V_original = app.V;
  igl::adjacency_list(app.F, app.A);
  
  // 2. Initialisation
  init_fixed_zone(app); // Fixe le bas automatiquement

  // 3. Setup Viewer
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(app.V, app.F);
  viewer.data().point_size = 8;

  std::cout << "=== MODE D'EMPLOI ===\n";
  std::cout << "  [SOURIS] Clic Gauche : Selectionner Poignee (Rouge)\n";
  std::cout << "  [PAVE NUM] +/- : Taille du pinceau\n";
  std::cout << "  [C] : Effacer la selection Poignee\n";
  std::cout << "  [ESPACE] : CALCULER les poids (Degrade)\n";
  std::cout << "  [FLECHES] : DEFORMER le maillage\n";
  std::cout << "  [J] : Changer mode de transfert (Lineaire/Smooth/Carre)\n";
  std::cout << "  [R] : Reset\n";

  // 4. Callback Clavier
  viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& v, unsigned int key, int) {
    bool update_view = false;
    
    // Outils de sélection
    if (key == KEY_PLUS) { 
        app.k++; update_view=true; 
    }
    else if (key == KEY_MINUS && app.k > 0) { 
        app.k--; update_view=true; 
    }
    else if (key == KEY_C) { 
        app.handle_vertices.clear(); 
        std::cout << "Selection effacee.\n"; 
        update_view=true; 
    }
    
    // Calcul
    else if (key == KEY_SPACE) { 
        compute_harmonic_weights(app); 
        update_view=true; 
    }
    
    // Déformation
    else if (key == KEY_UP || key == KEY_DOWN || key == KEY_LEFT || key == KEY_RIGHT) {
        float speed = 0.05f;
        if (key == KEY_UP)    
            app.translation.y() += speed;
        if (key == KEY_DOWN)  
            app.translation.y() -= speed;
        if (key == KEY_RIGHT) 
            app.translation.x() += speed;
        if (key == KEY_LEFT)  
            app.translation.x() -= speed;
        
        apply_deformation(app);
        v.data().set_vertices(app.V);
        v.data().compute_normals();
    }
    
    // Options
    else if (key == KEY_J) {
        app.transfer_function = (app.transfer_function + 1) % 3;
        std::string modes[] = {"Lineaire", "Smoothstep", "Carre"};
        std::cout << "Mode Transfert : " << modes[app.transfer_function] << std::endl;
        if(app.translation.norm() > 0) { // Ré-appliquer si déjà déformé
            apply_deformation(app);
            v.data().set_vertices(app.V);
            v.data().compute_normals();
        }
    }
    else if (key == KEY_R) { 
        app.V = app.V_original; 
        app.translation.setZero(); 
        app.handle_vertices.clear(); 
        app.weights_computed = false; 
        v.data().set_vertices(app.V); 
        v.data().compute_normals();
        update_view=true; 
    }

    if(update_view) update_colors(app, v);
    return false;
  };

  // 5. Callback Souris
  viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& v, int button, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      int fid; Eigen::Vector3f bc;
      // Raycasting pour trouver le sommet sous la souris
      if (igl::unproject_onto_mesh(Eigen::Vector2f(v.current_mouse_x, v.core().viewport(3) - v.current_mouse_y), 
                                   v.core().view, v.core().proj, v.core().viewport, app.V, app.F, fid, bc)){
        int vi; bc.maxCoeff(&vi); // Sommet le plus proche dans la face
        app.current_vertex = app.F(fid, vi);
        
        // Ajouter à la sélection (Poignée par défaut)
        std::set<int> region = get_neighbors(app, app.current_vertex);
        app.handle_vertices.insert(region.begin(), region.end());
        
        update_colors(app, v);
      }
    }
    return false;
  };

  update_colors(app, viewer);
  viewer.launch();
}

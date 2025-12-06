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
const unsigned int KEY_T = 84;      // Changer Fonction Transfert
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
    Eigen::VectorXd weights;       // Poids harmoniques (0..1)
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
    double epsilon = 0.05 * (max_y - min_y); // 5% de tolérance

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
        app.C = Eigen::MatrixXd::Constant(app.V.rows(), 3, 0.8); // Gris par défaut
        
        for(int v : app.fixed_vertices)  app.C.row(v) << 0, 0, 1; // Bleu (Fixe)
        for(int v : app.handle_vertices) app.C.row(v) << 1, 0, 0; // Rouge (Poignée)
        
        // Prévisualisation du curseur (Vert)
        if(app.current_vertex != -1) {
            std::set<int> preview = get_neighbors(app, app.current_vertex);
            for(int v : preview) {
                // On n'écrase pas les couleurs si déjà sélectionné
                if (!app.fixed_vertices.count(v) && !app.handle_vertices.count(v))
                    app.C.row(v) << 0, 1, 0; 
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
        std::cout << "[ERREUR] Il faut une zone Fixe (Bleu) et une Poignee (Rouge) !\n";
        return;
    }

    std::cout << "[CALCUL] Resolution de l'equation de Laplace... ";
    
    int n = app.V.rows();
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(app.V_original, app.F, L); // Laplacien cotangente

    // Préparation des conditions aux limites (Dirichlet)
    // On fixe w=0 sur la zone fixe, et w=1 sur la poignée
    Eigen::VectorXi b(app.fixed_vertices.size() + app.handle_vertices.size()); // Indices contraints
    Eigen::VectorXd bc(b.size());                                              // Valeurs contraintes
    
    int idx = 0;
    for(int v : app.fixed_vertices)  { b(idx) = v; bc(idx) = 0.0; idx++; }
    for(int v : app.handle_vertices) { b(idx) = v; bc(idx) = 1.0; idx++; }
    
    // Résolution : min 0.5 w^T Q w  sous contrainte w(b) = bc
    // Q = -L car L est définie négative, et le solveur veut une matrice positive.
    Eigen::SparseMatrix<double> Q = -L;
    Eigen::VectorXd B = Eigen::VectorXd::Zero(n); // Terme linéaire (nul ici)
    Eigen::VectorXd Beq;                          // Contraintes d'égalité (vide)
    
    igl::min_quad_with_fixed_data<double> solver_data;
    if(!igl::min_quad_with_fixed_precompute(Q, b, Eigen::SparseMatrix<double>(), true, solver_data)) {
         std::cout << "ECHEC Precompute.\n"; return;
    }
    if(!igl::min_quad_with_fixed_solve(solver_data, B, bc, Beq, app.weights)) {
        std::cout << "ECHEC Solve.\n"; return;
    }

    app.weights_computed = true;
    std::cout << "OK !\n";
}

// ==========================================
// 4. DEFORMATION
// ==========================================

void apply_deformation(MeshApp& app) {
    if(!app.weights_computed) return;
    
    for(int i=0; i<app.V.rows(); ++i) {
        double w = app.weights(i); // Poids brut (0..1)
        double f_w = w;            // Poids transformé
        
        // Fonction de transfert
        if(app.transfer_function == 1)      f_w = w * w * (3 - 2 * w); // Smoothstep
        else if (app.transfer_function == 2) f_w = w * w;               // Squared
        
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
  std::cout << "  [T] : Changer mode de transfert (Lineaire/Smooth/Carre)\n";
  std::cout << "  [R] : Reset\n";

  // 4. Callback Clavier
  viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& v, unsigned int key, int) {
    bool update_view = false;
    
    // Outils de sélection
    if (key == KEY_PLUS) { app.k++; update_view=true; }
    else if (key == KEY_MINUS && app.k > 0) { app.k--; update_view=true; }
    else if (key == KEY_C) { app.handle_vertices.clear(); std::cout << "Selection effacee.\n"; update_view=true; }
    
    // Calcul
    else if (key == KEY_SPACE) { compute_harmonic_weights(app); update_view=true; }
    
    // Déformation
    else if (key == KEY_UP || key == KEY_DOWN || key == KEY_LEFT || key == KEY_RIGHT) {
        float speed = 0.05f;
        if (key == KEY_UP)    app.translation.y() += speed;
        if (key == KEY_DOWN)  app.translation.y() -= speed;
        if (key == KEY_RIGHT) app.translation.x() += speed;
        if (key == KEY_LEFT)  app.translation.x() -= speed;
        
        apply_deformation(app);
        v.data().set_vertices(app.V);
        v.data().compute_normals();
    }
    
    // Options
    else if (key == KEY_T) {
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
  std::cout << "  SPACE: Compute Weights (Wait for gradient)\n";
  std::cout << "  ARROWS: Move Handle\n";
  std::cout << "  C: Clear Handle Selection\n";
  std::cout << "  R: Reset\n";

  viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& v, unsigned int key, int) {
    // std::cout << "Key: " << key << std::endl;
    bool update = false;
    
    if (key == KEY_1) { app.selection_mode = 1; std::cout << "Mode: FIXED\n"; update=true; }
    else if (key == KEY_2) { app.selection_mode = 2; std::cout << "Mode: HANDLE\n"; update=true; }
    else if (key == KEY_PLUS) { app.k++; update=true; }
    else if (key == KEY_MINUS && app.k > 0) { app.k--; update=true; }
    else if (key == KEY_SPACE) { compute_harmonic_weights(app); update=true; }
    else if (key == 67) { // 'C' to clear handle
        app.handle_vertices.clear(); 
        std::cout << "Handle cleared.\n"; 
        update=true; 
    }
    else if (key == KEY_R) { 
        app.V = app.V_original; 
        app.translation.setZero(); 
        // Keep fixed vertices, but clear handle
        app.handle_vertices.clear(); 
        app.weights_computed = false; 
        app.selection_mode = 2; 
        v.data().set_vertices(app.V); 
        v.data().compute_normals();
        update=true; 
    }
    
    // Deformation keys
    float speed = 0.05f;
    if (key == KEY_UP) { app.translation.y() += speed; apply_deformation(app); v.data().set_vertices(app.V); v.data().compute_normals(); }
    if (key == KEY_DOWN) { app.translation.y() -= speed; apply_deformation(app); v.data().set_vertices(app.V); v.data().compute_normals(); }
    if (key == KEY_RIGHT) { app.translation.x() += speed; apply_deformation(app); v.data().set_vertices(app.V); v.data().compute_normals(); }
    if (key == KEY_LEFT) { app.translation.x() -= speed; apply_deformation(app); v.data().set_vertices(app.V); v.data().compute_normals(); }

    if(update) update_visualization(app, v);
    return false;
  };

  viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& v, int button, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      int fid;
      Eigen::Vector3f bc;
      double x = v.current_mouse_x;
      double y = v.core().viewport(3) - v.current_mouse_y;
      if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), v.core().view, v.core().proj, v.core().viewport, app.V, app.F, fid, bc)){
        int vi;
        bc.maxCoeff(&vi);
        app.current_vertex = app.F(fid, vi);
        
        // Apply selection if mode is active
        if (app.selection_mode > 0) {
            std::set<int> region = get_k_ring(app.A, app.current_vertex, app.k);
            if (app.selection_mode == 1) app.fixed_vertices.insert(region.begin(), region.end());
            if (app.selection_mode == 2) app.handle_vertices.insert(region.begin(), region.end());
        }
        update_visualization(app, v);
      }
    }
    return false;
  };

  update_visualization(app, viewer);
  viewer.launch();
}

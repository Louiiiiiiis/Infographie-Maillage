#include <igl/readOFF.h>
#include <igl/adjacency_list.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <queue>
#include <set>

const unsigned int KEY__PLUS = 334; //touche + pour k+1anneaux
const unsigned int KEY__MOINS = 333; //touche - pour k-1 anneaux
const unsigned int KEY_L = 76; // L pour Laplacien Explicit
const unsigned int KEY_M = 77; // M pour Laplacien Implicit
const unsigned int KEY_R = 82; // R pour Reset
const unsigned int KEY_C = 67; // C pour Couleur Laplacien

struct MeshApp {
    Eigen::MatrixXd V, V_original, C;
    Eigen::MatrixXi F;
    std::vector<std::vector<int>> A; //A[i] est la liste des indices des sommets adjacents au sommet i.
    std::set<int> visited;
    int sommet = 0;
    int k = 0;
};
//-------------------------------------------------

std::set<int> k_anneaux(const std::vector<std::vector<int>>& A, int v0, int k) {
    std::set<int> visited;
    std::queue<std::pair<int, int>> q;
    q.push({v0, 0});
    visited.insert(v0);

    while(!q.empty()) {
        auto [v, depth] = q.front(); q.pop();
        if (depth >= k) continue;
        for (int n : A[v]) {
            if (!visited.count(n)) {
                visited.insert(n);
                q.push({n, depth + 1});
            }
        }
    }
    return visited;
}
//fonction update couleur des k-anneaux
void update_colors(MeshApp& app, igl::opengl::glfw::Viewer& viewer) {
    app.visited = k_anneaux(app.A, app.sommet, app.k);
    app.C = Eigen::MatrixXd::Constant(app.V.rows(), 3, 0.8); // gris clair
    for (int i : app.visited)
        app.C.row(i) << 1, 0, 0;
    viewer.data().set_colors(app.C); //colore le maillage
}

void update_laplacian_colors(MeshApp& app, igl::opengl::glfw::Viewer& viewer) {
    Eigen::SparseMatrix<double> L, M, Minv;
    igl::cotmatrix(app.V, app.F, L);
    igl::massmatrix(app.V, app.F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
    igl::invert_diag(M, Minv);
    
    Eigen::MatrixXd HN = Minv * (L * app.V);
    Eigen::VectorXd H = HN.rowwise().norm(); // Magnitude of Mean Curvature
    
    igl::jet(H, true, app.C);
    viewer.data().set_colors(app.C);
}

//-------------------------------------------------

void LaplacienStepDiff(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &nouveauV, double lambda = 0.00001) {
  Eigen::SparseMatrix<double> C;
  igl::cotmatrix(V, F, C);
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

  Eigen::SparseMatrix<double> Minv;
  igl::invert_diag(M, Minv);
  Eigen::SparseMatrix<double> Lnorm = Minv * C;

  // Maj
  nouveauV = V + lambda * (Lnorm*V);
}

void LaplacienStepSL(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &nouveauV, double lambda = 0.0001) {
  Eigen::SparseMatrix<double> C;
  igl::cotmatrix(V, F, C);
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

  Eigen::SparseMatrix<double> Q=M - lambda *C;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(Q);
  Eigen::MatrixXd RHS=M*V;
  nouveauV=solver.solve(RHS);
}

int main(int argc, char *argv[]){
  MeshApp app;
  igl::readOFF("../meshes/bunny.off", app.V, app.F); // chargement mesh format off
  app.V_original = app.V; // Sauvegarde pour reset
  igl::adjacency_list(app.F, app.A);
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(app.V, app.F);
  viewer.data().point_size = 5;

  // callback clavier
  viewer.callback_key_down = [&](igl::opengl::glfw::Viewer&, unsigned int key, int) {
    std::cout << "Key pressed: " << key << std::endl;
    if (key == KEY__PLUS) { 
      app.k++; 
      update_colors(app, viewer); 
    }
    else if (key == KEY__MOINS && app.k >= 1) { 
      app.k--; 
      update_colors(app, viewer); 
    }
    else if (key == KEY_L) {
      Eigen::MatrixXd nouveauV;
      LaplacienStepDiff(app.V, app.F, nouveauV);
      app.V=nouveauV;
      viewer.data().set_vertices(app.V);
      viewer.data().compute_normals();
    }
    else if (key == KEY_M) {
      Eigen::MatrixXd nouveauV;
      LaplacienStepSL(app.V, app.F, nouveauV);
      app.V=nouveauV;
      viewer.data().set_vertices(app.V);
      viewer.data().compute_normals();
    }
    else if (key == KEY_R) {
        app.V = app.V_original;
        viewer.data().set_vertices(app.V);
        viewer.data().compute_normals();
        update_colors(app, viewer); // Reset colors too
    }
    else if (key == KEY_C) {
        update_laplacian_colors(app, viewer);
    }
    return false;
  };
  // callback souris
  viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& v, int boutton, int modifier) {
    if (boutton == GLFW_MOUSE_BUTTON_LEFT) {
      int fid;            // index 
      Eigen::Vector3f bc; // coordonnées barycentriques
      double x = v.current_mouse_x;
      double y = v.core().viewport(3) - v.current_mouse_y; // conversion coords écran
      if (igl::unproject_onto_mesh( Eigen::Vector2f(x, y), v.core().view, v.core().proj, v.core().viewport, app.V, app.F, fid, bc)){
        // Trouver le sommet le plus proche dans la face cliquée
        int vi;
        bc.maxCoeff(&vi);
        app.sommet = app.F(fid, vi);
        update_colors(app, viewer);
      }
    }
    return false;
  };

  update_colors(app, viewer);
  viewer.launch();
}

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
#include <iostream>

// Keys
const unsigned int KEY_PLUS = 334;  // Numpad +
const unsigned int KEY_MINUS = 333; // Numpad -
const unsigned int KEY_1 = 49;      // Select Fixed
const unsigned int KEY_2 = 50;      // Select Handle
const unsigned int KEY_SPACE = 32;  // Compute Weights
const unsigned int KEY_UP = 265;
const unsigned int KEY_DOWN = 264;
const unsigned int KEY_LEFT = 263;
const unsigned int KEY_RIGHT = 262;
const unsigned int KEY_R = 82;      // Reset

struct MeshApp {
    Eigen::MatrixXd V, V_original, C;
    Eigen::MatrixXi F;
    std::vector<std::vector<int>> A;
    
    // Selection
    int k = 0;
    int current_vertex = -1;
    int selection_mode = 0; // 0 = None, 1 = Fixed, 2 = Handle
    std::set<int> fixed_vertices;
    std::set<int> handle_vertices;
    
    // Deformation
    Eigen::VectorXd weights;
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    bool weights_computed = false;
};

// Helper: Get K-Ring
std::set<int> get_k_ring(const std::vector<std::vector<int>>& A, int v0, int k) {
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

// Update Colors based on state
void update_visualization(MeshApp& app, igl::opengl::glfw::Viewer& viewer) {
    // Default: Grey
    if(app.C.rows() != app.V.rows()) {
        app.C = Eigen::MatrixXd::Constant(app.V.rows(), 3, 0.8);
    }
    
    // If weights computed, show heat map
    if(app.weights_computed && app.weights.size() == app.V.rows()) {
        igl::jet(app.weights, true, app.C);
    } else {
        // Reset to grey
        app.C = Eigen::MatrixXd::Constant(app.V.rows(), 3, 0.8);
        
        // Show Fixed (Blue)
        for(int v : app.fixed_vertices) app.C.row(v) << 0, 0, 1;
        
        // Show Handle (Red)
        for(int v : app.handle_vertices) app.C.row(v) << 1, 0, 0;
        
        // Show Current Selection Preview (Green)
        if(app.current_vertex != -1) {
            std::set<int> preview = get_k_ring(app.A, app.current_vertex, app.k);
            for(int v : preview) {
                // Don't overwrite if already fixed/handle unless we are editing that set
                bool is_fixed = app.fixed_vertices.count(v);
                bool is_handle = app.handle_vertices.count(v);
                
                if (app.selection_mode == 1) app.C.row(v) << 0, 0.5, 1; // Light Blue
                else if (app.selection_mode == 2) app.C.row(v) << 1, 0.5, 0; // Orange
                else app.C.row(v) << 0, 1, 0; // Green (Just inspecting)
            }
        }
    }
    viewer.data().set_colors(app.C);
}

// Compute Harmonic Weights
void compute_harmonic_weights(MeshApp& app) {
    if(app.fixed_vertices.empty() || app.handle_vertices.empty()) {
        std::cout << "Need both Fixed and Handle vertices!" << std::endl;
        return;
    }

    int n = app.V.rows();
    Eigen::SparseMatrix<double> L, M;
    igl::cotmatrix(app.V_original, app.F, L); // Use original geometry for Laplacian
    
    // Boundary Conditions
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    std::vector<int> boundary_indices;
    
    for(int v : app.fixed_vertices) {
        boundary_indices.push_back(v);
        b(v) = 0.0;
    }
    for(int v : app.handle_vertices) {
        boundary_indices.push_back(v);
        b(v) = 1.0;
    }
    
    // Modify system for boundary conditions (L w = 0 s.t. constraints)
    // We want to solve L*w = 0, but enforce w[fixed]=0, w[handle]=1.
    // A simple way is to replace rows of L with Identity and RHS with value.
    // Or use min_quad_with_fixed.
    // Let's do the "replace row" method for simplicity or min_quad.
    // Actually, min_quad_with_fixed is standard in libigl but requires extra include.
    // Let's implement the row replacement manually on L.
    
    Eigen::SparseMatrix<double> A = L;
    for(int idx : boundary_indices) {
        // Zero out the row
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, idx); it; ++it) {
            it.valueRef() = 0;
        }
        // Set diagonal to 1
        A.coeffRef(idx, idx) = 1.0;
        // Note: This makes the matrix non-symmetric, SimplicialLLT might fail.
        // Use LU or BiCGSTAB. Or keep symmetry by zeroing columns too (but RHS changes).
        // Let's use SparseLU.
    }
    
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if(solver.info() != Eigen::Success) {
        std::cout << "Solver failed" << std::endl;
        return;
    }
    app.weights = solver.solve(b);
    app.weights_computed = true;
    std::cout << "Weights computed!" << std::endl;
}

// Apply Deformation
void apply_deformation(MeshApp& app) {
    if(!app.weights_computed) return;
    
    for(int i=0; i<app.V.rows(); ++i) {
        double w = app.weights(i);
        // Smoothstep for nicer falloff? f(w) = w*w*(3-2w)
        // double f_w = w * w * (3 - 2 * w); 
        double f_w = w; // Linear for now
        
        app.V.row(i) = app.V_original.row(i) + f_w * app.translation.cast<double>().transpose();
    }
}

int main(int argc, char *argv[]){
  MeshApp app;
  igl::readOFF("../meshes/bunny.off", app.V, app.F);
  app.V_original = app.V;
  igl::adjacency_list(app.F, app.A);
  
  // Auto-select Fixed vertices (Bottom of the bunny)
  double min_y = app.V.col(1).minCoeff();
  double epsilon = 0.02 * (app.V.col(1).maxCoeff() - min_y); // 2% tolerance
  for(int i=0; i<app.V.rows(); ++i) {
      if(app.V(i, 1) < min_y + epsilon) {
          app.fixed_vertices.insert(i);
      }
  }
  std::cout << "Auto-selected " << app.fixed_vertices.size() << " fixed vertices (Bottom).\n";
  
  // Default to Handle selection mode
  app.selection_mode = 2; 
  app.k = 2; // Default K-ring size slightly larger

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(app.V, app.F);
  viewer.data().point_size = 8;

  std::cout << "COMMANDS:\n";
  std::cout << "  [AUTO] Bottom is Fixed (Blue)\n";
  std::cout << "  [DEFAULT] Click to select Handle (Red)\n";
  std::cout << "  +/-: Adjust brush size\n";
  std::cout << "  SPACE: Compute Weights (Wait for gradient)\n";
  std::cout << "  ARROWS: Move Handle\n";
  std::cout << "  R: Reset\n";

  viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& v, unsigned int key, int) {
    // std::cout << "Key: " << key << std::endl;
    bool update = false;
    
    if (key == KEY_1) { app.selection_mode = 1; std::cout << "Mode: FIXED\n"; update=true; }
    else if (key == KEY_2) { app.selection_mode = 2; std::cout << "Mode: HANDLE\n"; update=true; }
    else if (key == KEY_PLUS) { app.k++; update=true; }
    else if (key == KEY_MINUS && app.k > 0) { app.k--; update=true; }
    else if (key == KEY_SPACE) { compute_harmonic_weights(app); update=true; }
    else if (key == KEY_R) { 
        app.V = app.V_original; 
        app.translation.setZero(); 
        // Keep fixed vertices on reset for convenience? Or clear? 
        // Let's keep them to save time.
        app.handle_vertices.clear(); 
        app.weights_computed = false; 
        app.selection_mode = 2; // Back to handle select
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

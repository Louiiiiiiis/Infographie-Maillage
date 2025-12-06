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
    
    // Prepare Boundary Conditions
    Eigen::VectorXi b(app.fixed_vertices.size() + app.handle_vertices.size());
    Eigen::VectorXd bc(b.size());
    
    int idx = 0;
    for(int v : app.fixed_vertices) {
        b(idx) = v;
        bc(idx) = 0.0;
        idx++;
    }
    for(int v : app.handle_vertices) {
        b(idx) = v;
        bc(idx) = 1.0;
        idx++;
    }
    
    // Solve min 0.5 w^T L w subject to w(b) = bc
    // This solves Laplace equation with Dirichlet BC
    // L is negative semi-definite, so we minimize w^T (-L) w
    Eigen::SparseMatrix<double> Q = -L;
    
    Eigen::VectorXd B = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd Beq; // Empty linear equality constraints
    igl::min_quad_with_fixed_data<double> mqwf;
    // Linear term is 0
    if(!igl::min_quad_with_fixed_precompute(Q, b, Eigen::SparseMatrix<double>(), true, mqwf)) {
         std::cout << "Precompute failed" << std::endl;
         return;
    }
    
    if(!igl::min_quad_with_fixed_solve(mqwf, B, bc, Beq, app.weights)) {
        std::cout << "Solve failed" << std::endl;
        return;
    }

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
  double max_y = app.V.col(1).maxCoeff();
  double epsilon = 0.05 * (max_y - min_y); // Increased to 5% tolerance
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

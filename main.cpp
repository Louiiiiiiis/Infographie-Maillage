#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/adjacency_list.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>
#include <igl/read_triangle_mesh.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <set>

// Global variables for mesh and processing
Eigen::MatrixXd V, V_original;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> L, M;
std::vector<std::vector<int>> adj;
int selected_vertex = -1;
int k_ring_size = 1;
bool show_laplacian_color = false;

// Helper to get K-Ring neighbors
std::vector<int> get_k_ring(int start_vertex, int k, const std::vector<std::vector<int>>& adjacency) {
    std::set<int> visited;
    std::set<int> current_ring;
    std::set<int> next_ring;
    
    current_ring.insert(start_vertex);
    visited.insert(start_vertex);
    
    for(int i = 0; i < k; ++i) {
        next_ring.clear();
        for(int v : current_ring) {
            for(int neighbor : adjacency[v]) {
                if(visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    next_ring.insert(neighbor);
                }
            }
        }
        if(next_ring.empty()) break;
        current_ring = next_ring;
    }
    
    return std::vector<int>(visited.begin(), visited.end());
}

// Compute Laplacian Magnitude for coloring
void update_laplacian_color(igl::opengl::glfw::Viewer& viewer) {
    if (L.rows() == 0 || M.rows() == 0) return;

    // L * V gives the Laplacian vectors (approx mean curvature normals)
    // We want the magnitude of M^-1 * L * V for the scalar value
    Eigen::SparseMatrix<double> Minv;
    igl::invert_diag(M, Minv);
    Eigen::MatrixXd LV = Minv * (L * V);
    Eigen::VectorXd H = LV.rowwise().norm(); // Magnitude

    Eigen::MatrixXd C;
    igl::jet(H, true, C);
    viewer.data().set_colors(C);
}

// Explicit Smoothing (Diffusion)
void smooth_explicit(double dt = 0.01) {
    if (L.rows() == 0) return;
    // V_new = V + dt * M^-1 * L * V
    // Note: L in libigl is usually negative semi-definite (cotan), so diffusion is + L
    // But check sign: Energy E = 0.5 V^T L V. Gradient is L V. Flow is -Gradient.
    // So dV/dt = - (M^-1 * L * V). 
    // Let's try positive first, if it explodes or sharpens, we flip.
    // Standard cotmatrix is negative semi-definite. So L*V points "inwards" (curvature).
    // To smooth, we want to move in direction of curvature? No, mean curvature flow moves in direction of mean curvature vector.
    // Mean curvature vector H = -Laplace(V). 
    // So dV/dt = H = -Laplace(V).
    // If L is negative semi-definite, then -L is positive semi-definite.
    // Actually, let's just use the standard flow: V = V + dt * (M^-1 * L * V)
    
    Eigen::SparseMatrix<double> Minv;
    igl::invert_diag(M, Minv);
    V = V + dt * (Minv * (L * V));
}

// Implicit Smoothing (Linear System)
void smooth_implicit(double dt = 0.01) {
    if (L.rows() == 0) return;
    // (M - dt * L) * V_new = M * V
    // Again, check sign. If Explicit is V + dt*Minv*L*V, then Implicit is (I - dt*Minv*L)V_new = V
    // => (M - dt*L) V_new = M V
    
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> A = M - dt * L;
    solver.compute(A);
    if(solver.info() != Eigen::Success) {
        std::cout << "Decomposition failed!" << std::endl;
        return;
    }
    V = solver.solve(M * V);
}

int main(int argc, char *argv[])
{
    // Load mesh if provided, otherwise use cube
    if (argc > 1) {
        igl::read_triangle_mesh(argv[1], V, F);
    } else {
        // Inline mesh of a cube
        V = (Eigen::MatrixXd(8,3)<<
        0.0,0.0,0.0,
        0.0,0.0,1.0,
        0.0,1.0,0.0,
        0.0,1.0,1.0,
        1.0,0.0,0.0,
        1.0,0.0,1.0,
        1.0,1.0,0.0,
        1.0,1.0,1.0).finished();
        F = (Eigen::MatrixXi(12,3)<<
        0,6,4,
        0,2,6,
        0,3,2,
        0,1,3,
        2,7,6,
        2,3,7,
        4,6,7,
        4,7,5,
        0,4,5,
        0,5,1,
        1,5,7,
        1,7,3).finished();
    }
    
    V_original = V;

    // Compute Adjacency
    igl::adjacency_list(F, adj);

    // Compute Laplacian and Mass matrix
    igl::cotmatrix(V, F, L);
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(false); // Changed to false to see smooth shading if needed
    viewer.data().point_size = 8.0f;
    viewer.data().show_overlay = true;

    // Helper to update visualization of selection
    auto update_selection_viz = [&](igl::opengl::glfw::Viewer& v) {
        if (selected_vertex == -1) return;
        
        std::cout << "Updating selection for Vertex: " << selected_vertex << " (K=" << k_ring_size << ")" << std::endl;
        
        // Compute K-Ring
        std::vector<int> neighbors = get_k_ring(selected_vertex, k_ring_size, adj);
        
        // Visualize Selection
        v.data().clear_points();
        // Center point red
        v.data().add_points(V.row(selected_vertex), Eigen::RowVector3d(1,0,0));
        // Neighbors green
        for(int n : neighbors) {
            if(n != selected_vertex)
                v.data().add_points(V.row(n), Eigen::RowVector3d(0,1,0));
        }
    };

    // Mouse callback for selection
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& v, int b, int)->bool{
        if (b != GLFW_MOUSE_BUTTON_MIDDLE)
            return false;

        const auto& c = v.core();
        Eigen::Vector2f m(v.current_mouse_x, c.viewport(3) - v.current_mouse_y);

        float best = std::numeric_limits<float>::infinity();
        Eigen::Vector3f w;
        int closest_v = -1;
        
        for (int i = 0; i < V.rows(); ++i) {
            igl::project(V.row(i).cast<float>(), c.view, c.proj, c.viewport, w);
            float d2 = (w.head<2>() - m).squaredNorm();
            if (d2 < best) {
                best = d2;
                closest_v = i;
            }
        }

        if (closest_v >= 0) {
            selected_vertex = closest_v;
            update_selection_viz(v);
            return true;
        }
        return false;
    };

    // Keyboard callback
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& v, unsigned int key, int modifiers)->bool {
        // std::cout << "Key pressed: " << key << std::endl; // Debug
        switch(key) {
            case '+':
            case '=':
            case 266: // GLFW_KEY_KP_ADD
                k_ring_size++;
                std::cout << "K-Ring Size: " << k_ring_size << std::endl;
                update_selection_viz(v);
                return true;
            case '-':
            case '_':
            case 265: // GLFW_KEY_KP_SUBTRACT
                if(k_ring_size > 0) k_ring_size--;
                std::cout << "K-Ring Size: " << k_ring_size << std::endl;
                update_selection_viz(v);
                return true;
            case 'D':
            case 'd':
                std::cout << "Explicit Smoothing (Diffusion)..." << std::endl;
                smooth_explicit(0.1); // Small step for explicit
                v.data().set_mesh(V, F);
                v.data().compute_normals();
                if(selected_vertex != -1) update_selection_viz(v); // Keep selection visible
                return true;
            case 'S':
            case 's':
                std::cout << "Implicit Smoothing (Linear System)..." << std::endl;
                smooth_implicit(1.0); // Larger step allowed for implicit
                v.data().set_mesh(V, F);
                v.data().compute_normals();
                if(selected_vertex != -1) update_selection_viz(v); // Keep selection visible
                return true;
            case 'R':
            case 'r':
                std::cout << "Resetting mesh..." << std::endl;
                V = V_original;
                v.data().set_mesh(V, F);
                v.data().compute_normals();
                if(selected_vertex != -1) update_selection_viz(v);
                return true;
            case 'L':
            case 'l':
                show_laplacian_color = !show_laplacian_color;
                if(show_laplacian_color) {
                    update_laplacian_color(v);
                } else {
                    v.data().set_colors(Eigen::RowVector3d(1,1,1)); // White
                }
                return true;
        }
        return false;
    };

    std::cout << "Usage:\n";
    std::cout << "  Middle Click: Select vertex and show K-Ring\n";
    std::cout << "  +/-: Increase/Decrease K-Ring size\n";
    std::cout << "  D: Explicit Smoothing (Diffusion)\n";
    std::cout << "  S: Implicit Smoothing (Linear System)\n";
    std::cout << "  L: Toggle Laplacian Magnitude Coloring\n";
    std::cout << "  R: Reset Mesh\n";

    viewer.launch();
}

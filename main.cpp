#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <algorithm>

int main(int argc, char *argv[])
{
    int selected = -1;

    // Inline mesh of a cube
    const Eigen::MatrixXd V= (Eigen::MatrixXd(8,3)<<
    0.0,0.0,0.0,
    0.0,0.0,1.0,
    0.0,1.0,0.0,
    0.0,1.0,1.0,
    1.0,0.0,0.0,
    1.0,0.0,1.0,
    1.0,1.0,0.0,
    1.0,1.0,1.0).finished();
    const Eigen::MatrixXi F = (Eigen::MatrixXi(12,3)<<
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


    
    //Chope tous les voisins directs
    std::vector<std::vector<int>> adj(V.rows());
    for (int t = 0; t < F.rows(); ++t) {
      int a = F(t,0), b = F(t,1), c = F(t,2);
      adj[a].push_back(b); adj[b].push_back(a);
      adj[a].push_back(c); adj[c].push_back(a);
      adj[b].push_back(c); adj[c].push_back(b);
    }
    // dédoublonne proprement
    for (auto& nb : adj) {
      std::sort(nb.begin(), nb.end());
      nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }



    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.data().point_size = 8.0f;
    viewer.data().show_overlay = true;

    int sommet_select = -1;
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& v, int b, int)->bool{
      if (b != GLFW_MOUSE_BUTTON_MIDDLE)
        return false;

      const auto& c = v.core();
      Eigen::Vector2f m(v.current_mouse_x, c.viewport(3) - v.current_mouse_y);

      float best = std::numeric_limits<float>::infinity();
      Eigen::Vector3f w;
      for (int i = 0; i < V.rows(); ++i) {
        igl::project(V.row(i).cast<float>(), c.view, c.proj, c.viewport, w);
        float d2 = (w.head<2>() - m).squaredNorm();
        if (d2 < best) {
            best = d2;
            sommet_select = i;
        }
      }
      if (sommet_select < 0)
        return false;
      selected = sommet_select;

      v.data().clear_points();
      v.data().add_points(V.row(sommet_select), Eigen::RowVector3d(1,0,0));
      std::cout << "Sommet sélectionné = " << selected << std::endl;

      std::vector<int> nb = adj[selected];

      std::cout << "Voisins du sommet " << selected << ": [";
      for (size_t k = 0; k < nb.size(); ++k) {
        std::cout << nb[k];
        if (k + 1 < nb.size()) std::cout << ", ";
      }
      std::cout << "]\n";

      return true;
    };



    
  

    viewer.launch();
}

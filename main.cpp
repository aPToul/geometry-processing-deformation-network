#include "biharmonic_precompute.h"
#include "biharmonic_solve.h"
#include "arap_precompute.h"
#include "arap_single_iteration.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/snap_points.h>
#include <igl/unproject_onto_mesh.h>
#include <Eigen/Core>
#include <iostream>
#include <stack>

using namespace std;

// Undoable
struct State
{
  // Rest and transformed control points
  Eigen::MatrixXd CV, CU;
  bool placing_handles = true;
} s;

int main(int argc, char *argv[])
{
  // Undo Management
  std::stack<State> undo_stack,redo_stack;
  const auto push_undo = [&](State & _s=s)
  {
    undo_stack.push(_s);
    // clear
    redo_stack = std::stack<State>();
  };
  const auto undo = [&]()
  {
    if(!undo_stack.empty())
    {
      redo_stack.push(s);
      s = undo_stack.top();
      undo_stack.pop();
    }
  };
  const auto redo = [&]()
  {
    if(!redo_stack.empty())
    {
      undo_stack.push(s);
      s = redo_stack.top();
      redo_stack.pop();
    }
  };

  // Empty matrix
  Eigen::MatrixXd empty;

  // Determines the type of Laplacian being used
  int mode = 0;

  // For network drawing
  int reset_network = 1;
  int community_number = 0;
  int labels_on = 0;

  Eigen::MatrixXd V, U;
  Eigen::MatrixXi E;
  long sel = -1;
  Eigen::RowVector3f last_mouse;
  igl::min_quad_with_fixed_data<double> biharmonic_data, arap_data;
  Eigen::SparseMatrix<double> arap_K;

  igl::viewer::Viewer viewer;

  // Loads Community
  const auto create_network = [&](int labels_on, int community_number, Eigen::MatrixXd & V_, Eigen::MatrixXi & E_, int reset_network)
  {
    viewer.data.clear();

    string root_folder = "../../redditCommunities/" + std::to_string(community_number + 1) + "/";

    // Load data
    if (reset_network == 1){
      cout << "Loading data" << endl;
      Eigen::MatrixXd V;
      // Load in points from file
      {
        Eigen::MatrixXd D;
        std::vector<std::vector<double> > vD;
        std::string line;
        std::fstream in;
        in.open(argc>1?argv[1]: root_folder +  "P.txt");
        while(in)
        {
          std::getline(in, line);
          std::vector<double> row;
          std::stringstream stream_line(line);
          double value;
          while(stream_line >> value) row.push_back(value);
          if(!row.empty()) vD.push_back(row);
        }
        igl::list_to_matrix(vD,D);
        assert(D.cols() == 3 && "Position file should have 3 columns");
        V = D.leftCols(3);
      }
      // cout << "Number of points: " << V.rows() << endl;

      Eigen::MatrixXi E;
      // Load in edges from file
      {
        Eigen::MatrixXi D;
        std::vector<std::vector<int> > vI;
        std::string line;
        std::fstream in;
        in.open(argc>1?argv[1]: root_folder + "E.txt");
        while(in)
        {
          std::getline(in, line);
          std::vector<int> row;
          std::stringstream stream_line(line);
          int value;
          while(stream_line >> value) row.push_back(value);
          if(!row.empty()) vI.push_back(row);
        }
        igl::list_to_matrix(vI,D);
        assert(D.cols() == 2 && "Edges file should have 2 columns");
        E = D.leftCols(2);
      }
      // cout << "Number of edges: " << E.rows() << endl;

      V_ = V;
      E_ = E;
    }

    // Add points
    // cout << "Setting points" << endl;
    Eigen::RowVector3d point_color = Eigen::RowVector3d(0, 0, 0);
    viewer.data.set_points(V_, point_color);

    // Add edges
    // cout << "Setting edges" << endl;
    Eigen::RowVector3d edge_color = Eigen::RowVector3d(0, 0, 0.5);
    viewer.data.set_edges(V_, E_, edge_color);

    if (labels_on == 1){
      // cout << "Setting labels" << endl;
      // Load in labels
      std::string line;
      std::fstream in;
      in.open(argc>1?argv[1]: root_folder + "L.txt");
      int label_count = 0;
      while(in)
      {
        std::getline(in, line);
        if (!line.empty()){
          // cout << V.row(label_count) << endl;
          // cout << line << endl;
          viewer.data.add_label(V_.row(label_count), line);
          label_count += 1;
        }
      }
      assert(V_.rows() == label_count && "Labels should be 1-1 with points");      
    }
  };

  reset_network = 1;
  create_network(labels_on, community_number, V, E, reset_network);
  U = V;

  std::cout<<R"(
C,c      [DEFAULT Community 1] Increment to next Community number.
L,l      [DEFAULT Off] Turn labels On/Off.
[click]  To place new control point
[drag]   To move control point
[space]  Toggle whether placing control points or deforming
M,m      Switch deformation methods
Q,q      Switch Laplacians for deformation
U,u      Update deformation (i.e., run another iteration of solver)
R,r      Reset control points 
⌘ Z      Undo
⌘ ⇧ Z    Redo
)";
  enum Method
  {
    BIHARMONIC = 0,
    ARAP = 1,
    NUM_METHODS = 2,
  } method = BIHARMONIC;

  const auto & update = [&]()
  {
    // predefined colors
    const Eigen::RowVector3d orange(1.0,0.7,0.2);
    const Eigen::RowVector3d yellow(1.0,0.9,0.2);
    const Eigen::RowVector3d blue(0.2,0.3,0.8);
    const Eigen::RowVector3d green(0.2,0.6,0.3);
    if(s.placing_handles)
    {
      // Show the network
      reset_network = 0;
      create_network(labels_on, community_number, V, E, reset_network);
      // Wiggle a little to make the selected point visible
      Eigen::MatrixXd show;
      if (s.CV.rows() > 0){      
        show = s.CV + 0.005 * Eigen::MatrixXd::Random(s.CV.rows(),s.CV.cols());
      }
      viewer.data.add_points(show, orange);
    }
    else
    {
      // SOLVE
      switch(method)
      {
        default:
        case BIHARMONIC:
        {
          Eigen::MatrixXd D;
          biharmonic_solve(biharmonic_data,s.CU-s.CV,D);
          U = V+D;
          break;
        }
        case ARAP:
        {
          arap_single_iteration(arap_data,arap_K,s.CU,U);
          break;
        }
      }

      // Show the updated network positions (U)
      reset_network = 0;
      create_network(labels_on, community_number, U, E, reset_network);

      // Wiggle a little to make the control point visible
      Eigen::MatrixXd show_U;
      if (s.CU.rows() > 0){      
        show_U = s.CU + 0.001 * Eigen::MatrixXd::Random(s.CU.rows(), s.CU.cols());
      }
      viewer.data.set_colors(method==BIHARMONIC?orange:yellow);
      viewer.data.add_points(show_U, method==BIHARMONIC?blue:green);
    }
  };

  viewer.callback_mouse_down = 
    [&](igl::viewer::Viewer&, int, int)->bool
  {
    last_mouse = Eigen::RowVector3f(
      viewer.current_mouse_x,viewer.core.viewport(3)-viewer.current_mouse_y,0);

    if(s.placing_handles)
    {
      Eigen::MatrixXf CP;
      igl::project(V,
        viewer.core.view * viewer.core.model, 
        viewer.core.proj, viewer.core.viewport, CP);
      Eigen::VectorXf D = (CP.rowwise()-last_mouse).rowwise().norm();
      sel = (D.minCoeff(&sel) < 30)?sel:-1;
      if(sel != -1)
      {
        Eigen::RowVector3d new_c = V.row(sel);
        if(s.CV.size()==0 || (s.CV.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
        {
          push_undo();
          s.CV.conservativeResize(s.CV.rows()+1,3);
          // Snap to closest vertex on hit face
          s.CV.row(s.CV.rows()-1) = new_c;
          update();
          return true;
        }
      }
    }
    else
    {
      // Move closest control point
      Eigen::MatrixXf CP;
      igl::project(
        Eigen::MatrixXf(s.CU.cast<float>()),
        viewer.core.view * viewer.core.model, 
        viewer.core.proj, viewer.core.viewport, CP);
      Eigen::VectorXf D = (CP.rowwise()-last_mouse).rowwise().norm();
      sel = (D.minCoeff(&sel) < 30)?sel:-1;
      if(sel != -1)
      {
        last_mouse(2) = CP(sel, 2);
        push_undo();
        update();
        return true;
      }
    }
    return false;
  };

  viewer.callback_mouse_move = [&](igl::viewer::Viewer &, int,int)->bool
  {
    if(sel!=-1)
    {
      Eigen::RowVector3f drag_mouse(
        viewer.current_mouse_x,
        viewer.core.viewport(3) - viewer.current_mouse_y,
        last_mouse(2));
      Eigen::RowVector3f drag_scene,last_scene;
      igl::unproject(
        drag_mouse,
        viewer.core.view*viewer.core.model,
        viewer.core.proj,
        viewer.core.viewport,
        drag_scene);
      igl::unproject(
        last_mouse,
        viewer.core.view*viewer.core.model,
        viewer.core.proj,
        viewer.core.viewport,
        last_scene);
      s.CU.row(sel) += (drag_scene-last_scene).cast<double>();
      last_mouse = drag_mouse;
      update();
      return true;
    }
    return false;
  };
  viewer.callback_mouse_up = [&](igl::viewer::Viewer&, int, int)->bool
  {
    sel = -1;
    return false;
  };
  viewer.callback_key_pressed = 
    [&](igl::viewer::Viewer &, unsigned int key, int mod)
  {
    switch(key)
    {
      case 'L':
      case 'l':
        labels_on = (labels_on + 1) % 2;
        reset_network = 0;
        create_network(labels_on, community_number, V, E, reset_network);
        return true;
      case 'c':
      case 'C':
        community_number = (community_number + 1) % 5;
        cout << "Community " << community_number + 1 << endl;
        reset_network = 1;
        // Will load a fresh network and correctly set V
        create_network(labels_on, community_number, V, E, reset_network);
        reset_network = 0;
        // Reset everything
        push_undo();
        s.CV = empty;
        s.CU = s.CV;
        return true;
      case 'M':
      case 'm':
      {
        method = (Method)(((int)(method)+1)%((int)(NUM_METHODS)));
        break;
      }
      case 'Q':
      case 'q':
      {
        // Only supports 2 modes
        mode = (mode + 1) % 2;

        Eigen::VectorXi b;
        igl::snap_points(s.CV,V,b);
        // PRECOMPUTATION
        biharmonic_precompute(V, E, b,biharmonic_data,mode);
        arap_precompute(V, E, b,arap_data,arap_K,mode);
        break;
      }
      case 'R':
      case 'r':
      {
        push_undo();
        s.CU = s.CV;
        break;
      }
      case 'U':
      case 'u':
      {
        // Just trigger an update
        break;
      }
      case ' ':
        push_undo();
        s.placing_handles ^= 1;
        if(!s.placing_handles && s.CV.rows()>0)
        {
          // Switching to deformation mode
          s.CU = s.CV;

          Eigen::VectorXi b;
          igl::snap_points(s.CV,V,b);
          // PRECOMPUTATION
          biharmonic_precompute(V,E,b,biharmonic_data,mode);
          arap_precompute(V,E,b,arap_data,arap_K,mode);
        }
        break;
      default:
        return false;
    }
    update();
    return true;
  };

  // Special callback for handling undo
  viewer.callback_key_down = 
    [&](igl::viewer::Viewer &, unsigned char key, int mod)->bool
  {
    if(key == 'Z' && (mod & GLFW_MOD_SUPER))
    {
      (mod & GLFW_MOD_SHIFT) ? redo() : undo();
      update();
      return true;
    }
    return false;
  };
  viewer.callback_pre_draw = 
    [&](igl::viewer::Viewer &)->bool
  {
    if(viewer.core.is_animating && !s.placing_handles && method == ARAP)
    {
      arap_single_iteration(arap_data,arap_K,s.CU,U);
      update();
    }
    return false;
  };

  //// Unsure what this does
  // viewer.core.show_lines = false;
  // viewer.data.face_based = true;

  viewer.core.is_animating = true;

  // Set data
  update();

  viewer.core.point_size = 17.5;
  viewer.core.line_width = 20.0f;
  viewer.launch();
  return EXIT_SUCCESS;
}

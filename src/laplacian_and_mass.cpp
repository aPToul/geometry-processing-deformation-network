#include "laplacian_and_mass.h"
#include <math.h>
#include <iostream>
#include <igl/edges.h>
#include <igl/cotmatrix.h>
#include "igl/massmatrix.h"

using namespace std;

void graph_Laplacian(
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double> & L)
{
  typedef Eigen::Triplet<double> T;

  std::vector<T> tripletList;
  // For each edge, two elements of L are filled in with + 1
  // We will add the diagonal elements after
  tripletList.reserve(E.rows() * 2);

  for(int edge_number = 0; edge_number < E.rows(); edge_number++)
  {
    auto start_node_index = E(edge_number, 0);
    auto end_node_index = E(edge_number, 1);
    
    tripletList.push_back(T(start_node_index, end_node_index, 1.0));
    tripletList.push_back(T(end_node_index, start_node_index, 1.0));
  }
  L.setFromTriplets(tripletList.begin(), tripletList.end());
  
  // Set up Laplacian equality: what leaves a node, enters it
  for (int diagIndex = 0; diagIndex < L.rows(); diagIndex++){
    L.coeffRef(diagIndex, diagIndex) = -1 * L.row(diagIndex).sum();
  }
}

void edge_weighted_Laplacian(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double> & L)
{
  // Unknown effect of added distance to all points
  double epsilon = 0.00; // 0.01 

  // Unknown effect of lower bounding the edge lengths
  double edge_threshold = 0.0000; // 0.0001

  // For debugging
  double all_edge_differences = 0;
  int edges_added = 0;
  // Slightly redundant since logic is now edge-wise
  // Will simply overwrite each half-edge double-visited
  for (int edgeIndex = 0; edgeIndex < E.rows(); edgeIndex++){
    auto edge = E.row(edgeIndex);
    int source = edge[0];
    int target = edge[1];
    double distance = (V.row(source)- V.row(target)).norm();

    if (distance < 0)
    {
      cout << "Assumption of positive side length violated!" << endl;
    }

    if (L.coeffRef(source, target) == 0)
    {
      if (distance > edge_threshold)
      {
        // Side 1 and 2
        L.coeffRef(source, target) = 1.0 / (distance + epsilon);
        L.coeffRef(target, source) = 1.0 / (distance + epsilon);
        all_edge_differences += distance;
        edges_added += 1;        
      }
    }
  }

  if (edge_threshold > 0){
    cout << "Edges added: " << edges_added << endl;    
  }

  if (epsilon > 0){
    cout << "Average edge length: " << all_edge_differences / edges_added << endl;
  }

  // Set up Laplacian equality: what leaves a node, enters it
  for (int diagIndex = 0; diagIndex < L.rows(); diagIndex++){
    // for some reason cannot actually edit the .diagonal()
    L.coeffRef(diagIndex, diagIndex) = -1 * L.row(diagIndex).sum();
  }
}

void laplacian_and_mass(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double> & L,
  Eigen::SparseMatrix<double> & M,
  int mode)
{
  if (mode == 0){
    // cout << "Graph Laplacian" << endl;
    graph_Laplacian(E, L);
    M.setIdentity();
  }
  if (mode == 1){
    // cout << "Weighted Laplacian" << endl;
    edge_weighted_Laplacian(V, E, L);
    M.setIdentity();
  }
}


#include "cotmatrix.h"
#include <math.h>
#include <iostream>

using namespace std;

void cotmatrix(
  const Eigen::MatrixXd & l,
  const Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double> & L)
{
  // Unknown effect of added distance to all points
  double epsilon = 0.00; 

  // Unknown effect of lower bounding the edge lengths
  double edge_threshold = 0.0000;

  // For debugging
  double all_edge_differences = 0;
  int edges_added = 0;
  // Slightly redundant since logic is now edge-wise
  // Will simply overwrite each half-edge double-visited
  for (int faceIndex = 0; faceIndex < F.rows(); faceIndex++){
    auto vertices = F.row(faceIndex);
    // Indices
    int v1 = vertices[0];
    int v2 = vertices[1];
    int v3 = vertices[2];

    // Lengths of [1,2],[2,0],[0,1]
    // Lengths of [v2, v3], [v3, v1], [v1 , v2]
    auto lengths = l.row(faceIndex);
    // Side lengths
    double s1 = lengths[0];
    double s2 = lengths[1];
    double s3 = lengths[2];

    if (s1 <= 0 or s2 <= 0 or s3 <= 0)
    {
    	cout << "Assumption of positive side length violated!" << endl;
    }

    if (L.coeffRef(v1, v2) == 0)
    {
      if (s3 > edge_threshold)
      {
        // Side 1 and 2
        L.coeffRef(v1, v2) = 1.0 / (s3 + epsilon);
        L.coeffRef(v2, v1) = 1.0 / (s3 + epsilon);
        all_edge_differences += s3;
        edges_added += 1;        
      }
    }

    if (L.coeffRef(v2, v3) == 0)
    {
      if (s1 > edge_threshold) 
      {
  	    // Side 2 and 3
  	    L.coeffRef(v2, v3) = 1.0 / (s1 + epsilon);
  	    L.coeffRef(v3, v2) = 1.0 / (s1 + epsilon);
  	    all_edge_differences += s1;
  	    edges_added += 1;
      }
    }

    if (L.coeffRef(v1, v3) == 0)
    {
      if (s2 > edge_threshold) 
      {
  	    // Side 3 and 1
  	    L.coeffRef(v1, v3) = 1.0 / (s2 + epsilon);
  	    L.coeffRef(v3, v1) = 1.0 / (s2 + epsilon);
  	    all_edge_differences += s2;
  	    edges_added += 1;
      }
    }
  }

  cout << "Edges added: " << edges_added << endl;
  cout << "Average edge length: " << all_edge_differences / edges_added << endl;

  // Set up Laplacian equality: what leaves a node, enters it
  for (int diagIndex = 0; diagIndex < L.rows(); diagIndex++){
    // for some reason cannot actually edit the .diagonal()
    L.coeffRef(diagIndex, diagIndex) = -1 * L.row(diagIndex).sum();
  }
}


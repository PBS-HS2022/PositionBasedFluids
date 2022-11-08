#ifndef GRID2_H
#define GRID2_H

#include <igl/colormap.h>
#include <Eigen/Core>
#include "Array2T.h"

class Grid2 {
public:
	Grid2() {}

	Grid2(int res_x, int res_y, double dx) {
		m_res_x = res_x;
		m_res_y = res_y;
		m_dx = dx;
		m_x = Array2d(res_x, res_y);
		buildMesh();
	}

	Array2d& x() { return m_x; }

	const Array2d& x() const { return m_x; }

	void buildMesh() {
		int num_vertices = (m_res_x + 1) * (m_res_y + 1);
		int num_faces = m_res_x * m_res_y * 2; // 2 triangles per cell

		m_V = Eigen::MatrixXd(num_vertices, 3);
		m_F = Eigen::MatrixXi(num_faces, 3);

		int i = 0;
		for (int y = 0; y <= m_res_y; ++y) {
			for (int x = 0; x <= m_res_x; ++x) {
				m_V.row(i++) = Eigen::RowVector3d(x, y, 0) * m_dx;
			}
		}

		i = 0;
		for (int y = 0; y < m_res_y; ++y) {
			for (int x = 0; x < m_res_x; ++x) {
				int vid = y * (m_res_x + 1) + x;
				int vid_right = vid + 1;
				int vid_right_up = vid_right + (m_res_x + 1);
				int vid_up = vid + (m_res_x + 1);
				m_F.row(i++) = Eigen::RowVector3i(vid, vid_right, vid_right_up);
				m_F.row(i++) = Eigen::RowVector3i(vid, vid_right_up, vid_up);				
			}
		}
	}
	
	void reset() {
		m_x.zero();
	}

	void applySource(double xmin, double xmax, double ymin, double ymax) {
		for (int y = (int)(ymin * m_res_y); y < (int)(ymax * m_res_y); y++) {
			for (int x = (int)(xmin * m_res_x); x < (int)(xmax * m_res_x); x++) {
				m_x(x, y) = 1.0;
			}
		}
	}

	void getMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const {
		V = m_V;
		F = m_F;
	}

	void getColors(Eigen::MatrixXd& C, bool normalize=false, bool faceColor=true) const { 
		if (faceColor) {
			if (C.rows() == 0) {
				int num_faces = m_res_x * m_res_y * 2; // 2 triangles per cell
				C = Eigen::MatrixXd(num_faces, 3);
			}
			int i = 0;
			double cmin = m_x(0, 0);
			double cmax = cmin;
			for (int y = 0; y < m_res_y; ++y) {
				for (int x = 0; x < m_res_x; ++x) {
					double c = m_x(x, y);
					if (normalize) {
						if (c > cmax) cmax = c;
						if (c < cmin) cmin = c;
					}
					else {
						C.row(i++).setConstant(c);
						C.row(i++).setConstant(c);
					}
				}
			}

			if (!normalize) return;
			else if (cmin == cmax) {
				C.setZero();
				return;
			}

			// std::cout << "cmin:" << cmin << " cmax:" << cmax << std::endl;		
			for (int y = 0; y < m_res_y; ++y) {
				for (int x = 0; x < m_res_x; ++x) {
					double c = m_x(x, y);
					c = (c - cmin) / (cmax - cmin); // [0,1]
					double r, g, b;
					igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, c, r, g, b);
					C.row(i++) = Eigen::RowVector3d(r, g, b);
					C.row(i++) = Eigen::RowVector3d(r, g, b);
				}
			}
		}
		else {
			// vertex color
			if (C.rows() == 0) {
				int num_vertices = (m_res_x + 1) * (m_res_y + 1);
				C = Eigen::MatrixXd(num_vertices, 3);
			}
			int i = 0;
			double cmin = m_x(0, 0);
			double cmax = cmin;
			for (int y = 0; y <= m_res_y; ++y) {
				for (int x = 0; x <= m_res_x; ++x) {
					int x0 = std::max(x - 1, 0);
					int x1 = std::min(x, m_res_x - 1);
					int y0 = std::max(y - 1, 0);
					int y1 = std::min(y, m_res_y - 1);

					double c00 = m_x(x0, y0);
					double c01 = m_x(x0, y1);
					double c10 = m_x(x1, y0);
					double c11 = m_x(x1, y1);
					double c = (c00 + c01 + c10 + c11) / 4;
					if (normalize) {
						if (c > cmax) cmax = c;
						if (c < cmin) cmin = c;
					}
					C.row(i++).setConstant(c);
				}
			}

			if (!normalize) return;
			else if (cmin == cmax) {
				C.setZero();
				return;
			}

			i = 0;
			// std::cout << "cmin:" << cmin << " cmax:" << cmax << std::endl;
			for (int y = 0; y <= m_res_y; ++y) {
				for (int x = 0; x <= m_res_x; ++x) {
					double c = (C(i, 0) - cmin) / (cmax - cmin); // [0,1]
					double r, g, b;
					igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, c, r, g, b);
					C.row(i++) = Eigen::RowVector3d(r, g, b);
				}
			}
		}
	}

	inline int getResolutionX() const {
		return m_res_x;
	}

	inline int getResolutionY() const {
		return m_res_y;
	}

	inline int getGridSpacing() const {
		return m_dx;
	}

protected:
	int m_res_x, m_res_y;
	double m_dx;
	Array2d m_x;
	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;
};

#endif // GRID2_H
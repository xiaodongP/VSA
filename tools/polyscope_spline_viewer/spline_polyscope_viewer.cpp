#include <polyscope/curve_network.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::array;
using std::string;
using std::vector;

struct ObjGeometry {
    vector<array<double, 3>> vertices;
    vector<array<size_t, 3>> triangles;
    vector<array<size_t, 2>> lines;
};

static bool file_exists(const string& filename) {
    std::ifstream fin(filename);
    return fin.good();
}

static bool ends_with(const string& s, const string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static string path_join(const string& a, const string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

static bool read_obj_geometry(const string& filename, ObjGeometry& geom) {
    geom = ObjGeometry();
    std::ifstream fin(filename);
    if (!fin.is_open()) return false;

    string line;
    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        string tag;
        ss >> tag;
        if (tag == "v") {
            array<double, 3> p{};
            ss >> p[0] >> p[1] >> p[2];
            geom.vertices.push_back(p);
        } else if (tag == "f") {
            array<size_t, 3> f{};
            for (int i = 0; i < 3; i++) {
                string token;
                ss >> token;
                size_t slash = token.find('/');
                if (slash != string::npos) token = token.substr(0, slash);
                f[i] = (size_t)std::stoul(token) - 1;
            }
            geom.triangles.push_back(f);
        } else if (tag == "l") {
            array<size_t, 2> e{};
            ss >> e[0] >> e[1];
            e[0]--;
            e[1]--;
            geom.lines.push_back(e);
        }
    }
    return !geom.vertices.empty();
}

static void register_curve_if_present(
    const string& name,
    const string& filename) {
    ObjGeometry geom;
    if (!file_exists(filename) || !read_obj_geometry(filename, geom)) {
        std::cout << "[skip] " << filename << "\n";
        return;
    }
    if (geom.lines.empty()) {
        std::cout << "[skip] " << filename << " has no line elements\n";
        return;
    }
    polyscope::registerCurveNetwork(name, geom.vertices, geom.lines);
    std::cout << "[curve] " << name << " vertices=" << geom.vertices.size()
              << " edges=" << geom.lines.size() << "\n";
}

static void register_surface_if_present(
    const string& name,
    const string& filename,
    const array<double, 3>& color,
    bool enabled) {
    ObjGeometry geom;
    if (!file_exists(filename) || !read_obj_geometry(filename, geom)) {
        std::cout << "[skip] " << filename << "\n";
        return;
    }
    if (geom.triangles.empty()) {
        std::cout << "[skip] " << filename << " has no triangle elements\n";
        return;
    }
    auto* surface = polyscope::registerSurfaceMesh(name, geom.vertices, geom.triangles);
    surface->setSmoothShade(true);
    surface->setSurfaceColor({
        (float)color[0],
        (float)color[1],
        (float)color[2]});
    surface->setEnabled(enabled);
    std::cout << "[surface] " << filename
              << " vertices=" << geom.vertices.size()
              << " faces=" << geom.triangles.size() << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: spline_polyscope_viewer <debug_prefix|pipeline_output_dir|mesh.obj>\n";
        std::cout << "Example: spline_polyscope_viewer interactive_spline_region_0\n";
        std::cout << "Example: spline_polyscope_viewer trimmed_bspline_output_region_1\n";
        std::cout << "Example: spline_polyscope_viewer trimmed_bspline_output_region_1/bspline_trimmed_surface_asset.obj\n";
        return 1;
    }

    string input = argv[1];
    string surface_file;
    string debug_prefix;
    string debug_dir;

    if (ends_with(input, ".obj") || ends_with(input, ".OBJ")) {
        surface_file = input;
    } else if (file_exists(input + "_sampled_surface.obj")) {
        debug_prefix = input;
        surface_file = input + "_sampled_surface.obj";
    } else if (file_exists(path_join(input, "bspline_trimmed_surface_asset.obj"))) {
        debug_dir = input;
        surface_file = path_join(input, "bspline_trimmed_surface_asset.obj");
    } else if (file_exists(path_join(input, "bspline_trimmed_surface.obj"))) {
        debug_dir = input;
        surface_file = path_join(input, "bspline_trimmed_surface.obj");
    } else {
        surface_file = input + "_sampled_surface.obj";
        debug_prefix = input;
    }

    ObjGeometry surface;
    if (!read_obj_geometry(surface_file, surface) || surface.triangles.empty()) {
        std::cerr << "Cannot read surface mesh: " << surface_file << "\n";
        return 2;
    }

    polyscope::init();
    polyscope::options::programName = "VSA B-spline Polyscope Viewer";

    auto* ps_surface = polyscope::registerSurfaceMesh(
        "trimmed B-spline asset surface",
        surface.vertices,
        surface.triangles);
    ps_surface->setSmoothShade(true);
    ps_surface->setSurfaceColor({0.40, 0.58, 0.95});

    std::cout << "[surface] " << surface_file
              << " vertices=" << surface.vertices.size()
              << " faces=" << surface.triangles.size() << "\n";

    if (!debug_dir.empty()) {
        string abc_file = path_join(debug_dir, "bspline_trimmed_surface_abc_preview.obj");
        if (file_exists(abc_file) && abc_file != surface_file) {
            ObjGeometry abc;
            if (read_obj_geometry(abc_file, abc) && !abc.triangles.empty()) {
                auto* abc_surface = polyscope::registerSurfaceMesh(
                    "ABC boundary-controlled preview",
                    abc.vertices,
                    abc.triangles);
                abc_surface->setSmoothShade(true);
                abc_surface->setSurfaceColor({0.95, 0.55, 0.20});
                abc_surface->setEnabled(false);
                std::cout << "[surface] " << abc_file
                          << " vertices=" << abc.vertices.size()
                          << " faces=" << abc.triangles.size() << "\n";
            }
        }
        register_surface_if_present(
            "ABC boundary ribbon strips",
            path_join(debug_dir, "abc_boundary_ribbon_strips.obj"),
            {0.20, 0.80, 0.45},
            false);
        register_surface_if_present(
            "ABC boundary ribbon B-spline surfaces",
            path_join(debug_dir, "abc_boundary_ribbon_surfaces.obj"),
            {0.10, 0.70, 0.82},
            false);
        register_curve_if_present(
            "asset control net",
            path_join(debug_dir, "trimmed_bspline_asset_control_net.obj"));
        register_curve_if_present(
            "asset UV trim loops",
            path_join(debug_dir, "trimmed_bspline_asset_trim_loops.obj"));
        register_curve_if_present(
            "asset surface trim loops",
            path_join(debug_dir, "trimmed_bspline_asset_surface_trim_loops.obj"));
    }
    if (!debug_prefix.empty()) {
        register_curve_if_present("control net", debug_prefix + "_control_net.obj");
        register_curve_if_present("boundary bottom", debug_prefix + "_boundary_bottom.obj");
        register_curve_if_present("boundary right", debug_prefix + "_boundary_right.obj");
        register_curve_if_present("boundary top", debug_prefix + "_boundary_top.obj");
        register_curve_if_present("boundary left", debug_prefix + "_boundary_left.obj");
    }

    polyscope::show();
    return 0;
}

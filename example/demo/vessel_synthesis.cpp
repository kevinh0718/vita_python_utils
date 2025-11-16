/**
*	Publication: Maso Talou, G.D., Safaei, S., Hunter, P.J. et al. Adaptive constrained constructive optimisation *   for complex vascularisation processes. Sci Rep 11, 6180 (2021). https://doi.org/10.1038/s41598-021-85434-9
*	@author: Gonzalo D Maso Talou
*	@affiliation: The University of Auckland
*/

// STD Libs
#include<cmath>
#include<string>
#include<ctime>
#include<cstdio>
#include<cstdlib>
#include<vector>

// VItA Libs
#include "structures/domain/AbstractDomain.h"
#include "structures/domain/StagedDomain.h"
#include "core/GeneratorData.h" 
#include "core/StagedFRROTreeGenerator.h"
#include "constrains/ConstantConstraintFunction.h"
#include "structures/tree/SingleVesselCCOOTree.h"
#include "structures/vascularElements/AbstractVascularElement.h"
#include "io/VTKObjectTreeNodalWriter.h"
#include "structures/domain/SimpleDomain.h"

using namespace std;

void vascularise(string output_filename, AbstractConstraintFunction<double, int>* gam,
    AbstractConstraintFunction<double, int>* delta, AbstractConstraintFunction<double, int>* eta,
    int n_draw, int seed, int N_fail, double l_lim_fr,
    double perfusion_area_factor, double close_neighborhood_factor, int Delta_nu,
    double theta_min, float xx, float yy, float zz,
    long long int n_term, string input_vtk)
{
    // Boundary conditions - Flow, radius and position of the inlet 
    double q0 {2};
    double r0 {0.01};
    point x0 {((float) xx), ((float) yy), ((float) zz)};
    
    // Optimisation parameters
    GeneratorData *gen_data_0 = new GeneratorData(16000, N_fail, l_lim_fr, perfusion_area_factor, close_neighborhood_factor, 0.25, Delta_nu, 0, false);    

    // Domain definition
    SimpleDomain *domain_0 = new SimpleDomain(input_vtk, n_draw, seed, gen_data_0);
    domain_0->setMinBifurcationAngle(theta_min);
    domain_0->setIsConvexDomain(false);

    // Define domain stages
    StagedDomain *staged_domain = new StagedDomain();
    staged_domain->addStage(n_term, domain_0);

    // Creation of DCCO generator
    StagedFRROTreeGenerator *tree_generator = new StagedFRROTreeGenerator(staged_domain, x0, r0, q0, n_term, {gam}, {delta}, {eta}, 0., 1.e-5);
    SingleVesselCCOOTree *tree = static_cast<SingleVesselCCOOTree *>(tree_generator->getTree());
    // Indicates that all meshes and boundary conditions are in cm (important for the viscosity computation).
    tree->setIsInCm(true);

    // Executes DCCO generator
    tree_generator->generate(10, ".");
        
    // Saves the outputs as CCO and VTP files
    tree->save(output_filename + ".cco");
    VTKObjectTreeNodalWriter *tree_writer = new VTKObjectTreeNodalWriter();
    tree_writer->write(output_filename + ".vtp", tree);
          
    delete tree_writer;
    delete tree_generator;
    delete staged_domain;
    delete domain_0;
    delete gen_data_0;
}


int main(int argc, char *argv[])
{
    // Consecutive attempts to generate a point - N_fail
    int N_fail = 200;
    // Correction step factor - fr - Eq (14)
    double l_lim_fr = 0.9;
    // Discretisation of the testing triangle for the bifurcation - Delta nu - Figure 1
    int Delta_nu = 7;

    // Geometrical constrains - 
    // Power-law coefficient - gamma - Eq (3) - Murray's law
    AbstractConstraintFunction<double,int> *gam {new ConstantConstraintFunction<double, int>(3.)};
    // Symmetry ratio parameter - delta - Eq (4)
    AbstractConstraintFunction<double,int> *delta {new ConstantConstraintFunction<double, int>(0)};
    // Viscosity in cP - eta - Eq (7)
    AbstractConstraintFunction<double,int> *eta {new ConstantConstraintFunction<double, int>(3.6)};

    // Buffer size for random point generation
    int n_draw {10000};
    // Minimum bifurcation angle - theta_min - Eq (17) 
    double theta_min {(3./18.) * M_PI};
    // l_min tuning parameter - nu - Eq (12)
    double perfusion_area_factor {0.001};
    // Heuristic parameter to reduce the neighbour vessels to be tested for connections in line 15 of Algorithm 2
    double close_neighbourhood_factor {4.0};
    // Number of terminals to be generated
    long long int n_term = 100;
    // Starting points in unit (cm)
    float xpos, ypos, zpos;
    // Random seed
    int seed {200}; //200 //250
    // Domain geometry
    string input_vtk;
    string path;
    if (argc == 8) {
        xpos = stof(argv[1]);
        ypos = stof(argv[2]);
        zpos = stof(argv[3]);
        n_term = stoi(argv[4]);
        path = argv[5];
        input_vtk = argv[6];
        seed = stoi(argv[7]);
    }
    else {
        xpos = 25;
        ypos = 25;
        zpos = 25;
        path = "demo_sphere";
        input_vtk = "sphere.vtk";
    }
    
    vascularise(path, gam, delta, eta, n_draw, seed, N_fail, l_lim_fr, perfusion_area_factor, close_neighbourhood_factor, Delta_nu, theta_min, xpos, ypos, zpos, n_term, input_vtk);
    
    delete gam;    
    delete delta;
    delete eta;
        
    return 0;
} 

/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"

struct Particle
{
    int id;
    double x;
    double y;
    double theta;
    double weight;
};

class ParticleFilter
{

    // Number of particles to draw
    int num_particles;

    // Flag, if filter is initialized
    bool is_initialized;

    // Vector of weights of all particles
    std::vector<double> weights;

public:

    // Set of current particles
    std::vector<Particle> particles;

    // Constructor
    // @param M Number of particles
    ParticleFilter()
        : num_particles(0), is_initialized(false)
    {
    }

    // Destructor
    ~ParticleFilter()
    {
    }

    void init(const double x, const double y, const double theta, const double std[]);

    void prediction(const double delta_t, const double std_pos[], const double velocity, const double yaw_rate);

    void CTRVPrediction(const double delta_t, const double velocity, const double yaw_rate, const double std_pos[], double &x, double &y, double &yaw);

    void dataAssociation(const std::vector<LandmarkObs> &predicted, std::vector<LandmarkObs> &observations);

    void updateWeights(const double sensor_range, const double std_landmark[], std::vector<LandmarkObs> &observations, const Map &map_landmarks);

    /**@brief calculate 2d gaussian probability;
     *
     * calculate 2d gaussian probability;
     * @param sigma_x [IN]: sigma x;
     * @param sigma_y [IN]: sigma y;
     * @param delta_x [IN]: delta x;
     * @param delta_y [IN]: delta y;
     * @return the probability;
     */
    double Gaussian2DProbability(const double sigma_x, const double sigma_y, const double delta_x, const double delta_y)
    {
        double sigma_x2 = sigma_x * sigma_x;
        double sigma_y2 = sigma_y * sigma_y;

        return exp(-0.5 * (delta_x * delta_x / sigma_x2 + delta_y * delta_y / sigma_y2)) / (sigma_x * sigma_y * 2 * M_PI);
    }

    void TransformMeasurements(const double x, const double y, const double yaw, const std::vector<LandmarkObs> &observations, std::vector<LandmarkObs> &transformed_obs);

    void resample();

    /*
     * write Writes particle positions to a file.
     * @param filename File to write particle positions to.
     */
    void write(std::string filename);

    /**
     * initialized Returns whether particle filter is initialized yet or not.
     */
    const bool initialized() const
    {
        return is_initialized;
    }
};


#endif /* PARTICLE_FILTER_H_ */

/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

//#define DEBUG_OUTPUT

/**@brief initialize particle filter;
 *
 * initialize particle filter by initializing particles to Gaussian distribution around first position and all the weights to 1.
 * @param x [IN]: Initial x position [m] (simulated estimate from GPS)
 * @param y [IN]: Initial y position [m]
 * @param theta [IN]: Initial orientation [rad]
 * @param std[] [IN]: Array of dimension 3 [standard deviation of x [m], standard deviation of y [m] standard deviation of yaw [rad]]
 */
void ParticleFilter::init(const double x, const double y, const double theta, const double std[])
{
    num_particles = 100;

    std::default_random_engine gen;
    std::normal_distribution<double> N_x(x, std[0]);
    std::normal_distribution<double> N_y(y, std[1]);
    std::normal_distribution<double> N_theta(theta, std[2]);

    Particle particle;
    for (int p = 0; p < num_particles; ++p)
    {
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight = 1.0;
        particle.id = p;

        particles.push_back(particle);
        weights.push_back(1.0);
    }
    is_initialized = true;

}

/**@brief predict a particle by CTRV model;
 *
 * predict a particle by CTRV model;
 * 1. predit pose by CTRV model;
 * 2. add noise to the pose;
 * @param delta_t [IN]: the time;
 * @param velocity [IN]: control velocity of the vehicle;
 * @param yaw_rate [IN]: control yaw rate of the vehicle;
 * @param std_pos [IN]: x, y, yaw standard deviation;
 * @param x [IN|OUT]: the pos x;
 * @param y [IN|OUT]: the pos y;
 * @param yaw [IN|OUT]: the pos yaw;
 */
void ParticleFilter::CTRVPrediction(const double delta_t, const double velocity, const double yaw_rate, const double *std_pos, double &x, double &y, double &yaw)
{
    //add noise to pose;
    static std::default_random_engine gen;/**@note should defined as static variable. if not static, the noise will be always the same. why ?*/
    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_yaw(0.0, std_pos[2]);

    double p_x = x;
    double p_y = y;
    double p_yaw = yaw;

    if (fabs(yaw_rate) > 1e-3)
    {
        p_x += velocity * (sin(yaw + yaw_rate * delta_t) - sin(yaw)) / yaw_rate;
        p_y += velocity * (-cos(yaw + yaw_rate * delta_t) + cos(yaw)) / yaw_rate;
    }
    else
    {
        p_x += velocity * cos(yaw) * delta_t;
        p_y += velocity * sin(yaw) * delta_t;
    }
    p_yaw += yaw_rate * delta_t;

    p_x += dist_x(gen);
    p_y += dist_y(gen);
    p_yaw += dist_yaw(gen);

    /**@todo normalize yaw ?*/

    x = p_x;
    y = p_y;
    yaw = p_yaw;
    return;
}

/**@brief predict the state for the next time step using the CTRV model;
 *
 * @param delta_t [IN]: Time between time step t and t+1 in measurements [s]
 * @param std_pos[] [IN]: Array of dimension 3 [standard deviation of x [m], standard deviation of y [m] standard deviation of yaw [rad]]
 * @param velocity [IN]: Velocity of car from t to t+1 [m/s]
 * @param yaw_rate [IN]: Yaw rate of car from t to t+1 [rad/s]
 */
void ParticleFilter::prediction(const double delta_t, const double std_pos[], const double velocity, const double yaw_rate)
{
#ifdef DEBUG_OUTPUT
    std::cout<<"----------------prediction---------------------\n";
#endif
    for (int i = 0; i < num_particles; ++i)
    {
#ifdef DEBUG_OUTPUT
        std::cout<<"particle "<<i<<" ("<<particles[i].x<<","<<particles[i].y<<","<<particles[i].theta<<") -->";
#endif
        CTRVPrediction(delta_t, velocity, yaw_rate, std_pos, particles[i].x, particles[i].y, particles[i].theta);

#ifdef DEBUG_OUTPUT
        std::cout<<" ("<<particles[i].x<<","<<particles[i].y<<","<<particles[i].theta<<")\n";
#endif
    }
    return;
}


/**@brief associate landmark measurements to landmarks in the map;
 *
 * finds which observations correspond to which landmarks (likely by using a nearest-neighbors data association).
 * @param predicted [IN]: the landmarks in the particle view range, in map coordinates system;
 * @param observations [IN|OUT]: the measurements of the landmarks by the particle, in map coordinates system;
 * @note the id of the observations is modified to the associated landmark id by this function;
 */
void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& landmarks, std::vector<LandmarkObs> &observations)
{
    for (int o = 0; o < observations.size(); ++o)
    {
        double min_dist = INFINITY;
        int min_id = 0;
        for (int l = 0; l < landmarks.size(); ++l)
        {
            double dist = (landmarks[l].x - observations[o].x) * (landmarks[l].x - observations[o].x) + (landmarks[l].y - observations[o].y) * (landmarks[l].y - observations[o].y);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_id = landmarks[l].id;
            }
        }

        observations[o].id = min_id;
    }
}

void ParticleFilter::TransformMeasurements(const double x, const double y, const double yaw, const std::vector<LandmarkObs> &observations, std::vector<LandmarkObs> &transformed_obs)
{
    transformed_obs.reserve(observations.size());
    for (int o = 0; o < observations.size(); ++o)
    {
        LandmarkObs obs;
        obs.id = observations[o].id;
        obs.x = observations[o].x * cos(yaw) - observations[o].y * sin(yaw) + x;
        obs.y = observations[o].x * sin(yaw) + observations[o].y * cos(yaw) + y;

        transformed_obs.push_back(obs);
    }
}

/**@brief update particle weights;
 *
 * Updates the weights for each particle based on the likelihood of the observed measurements.
 * for each particle:
 * 1. get all the landmarks in the view range (just used to accelerate the computation);
 * 2. transform all the landmark measurements into map frame;
 * 3. calculate the likelihood of the particle by multipling the likelihood of each measurement;
 * 4. normalize the weight (only used for easy debugging)
 * @param sensor_range [IN]: Range [m] of sensor
 * @param std_landmark[] [IN]: Array of dimension 2 [standard deviation of range [m], standard deviation of bearing [rad]]
 * @param observations Vector of landmark observations
 * @param map Map class containing map landmarks
 * @note The observations are given in the VEHICLE'S coordinate system. Your particles are located according to the MAP'S coordinate system. You will need to transform between the two systems. Keep in mind that this transformation requires both rotation AND translation (but no scaling). The following is a good resource for the theory: https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm and the following is a good resource for the actual equation to implement (look at equation 3.33. Note that you'll need to switch the minus sign in that equation to a plus to account for the fact that the map's y-axis actually points downwards.) http://planning.cs.uiuc.edu/node99.html
 */
void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[], std::vector<LandmarkObs>& observations, const Map& map_landmarks)
{
    double range_square = sensor_range * sensor_range;
    for (int i = 0; i < num_particles; ++i)
    {
        //get all the landmarks in the vehicle view range;
        std::vector<LandmarkObs> landmarks_in_range;
        for (int l = 0; l < map_landmarks.landmark_list.size(); ++l)
        {
            double dx = map_landmarks.landmark_list[l].x_f - particles[i].x;
            double dy = map_landmarks.landmark_list[l].y_f - particles[i].y;
            if (dx * dx + dy * dy < range_square)
            {
                LandmarkObs landmark;
                landmark.x = map_landmarks.landmark_list[l].x_f;
                landmark.y = map_landmarks.landmark_list[l].y_f;
                landmark.id = map_landmarks.landmark_list[l].id_i;

                landmarks_in_range.push_back(landmark);
            }
        }

        //transform the measured landmarks into map frame;
        std::vector<LandmarkObs> transformed_obs;
        TransformMeasurements(particles[i].x, particles[i].y, particles[i].theta, observations, transformed_obs);

        //associate the measured landmarks to landmarks in map;
        dataAssociation(landmarks_in_range, transformed_obs);

        //update weight;
        double total_prob = 1.0;
        for (int o = 0; o < transformed_obs.size(); ++o)
        {
            int idx = transformed_obs[o].id - 1;
            double delta_x = map_landmarks.landmark_list[idx].x_f - transformed_obs[o].x;
            double delta_y = map_landmarks.landmark_list[idx].y_f - transformed_obs[o].y;
            total_prob *= Gaussian2DProbability(std_landmark[0], std_landmark[1], delta_x, delta_y);
        }

        particles[i].weight = total_prob;
        weights[i] = total_prob;
    }

    //normalize weight;
    double sum = 0.0;
    for (int p = 0; p < num_particles; ++p)
    {
        sum += particles[p].weight;
    }

    for (int p = 0; p < num_particles; ++p)
    {
        particles[p].weight /= sum;
        weights[p] /= sum;
#ifdef DEBUG_OUTPUT
        std::cout << "particle idx: " << p << ", weight:" << weights[p] << std::endl;
#endif
    }

}

/**@brief resampling particles;
 *
 * resampling particles according to the weights to form the new set of particles;
 */
void ParticleFilter::resample()
{
    std::default_random_engine gen;
    std::discrete_distribution<int> dist(weights.begin(), weights.end());

    std::vector<Particle> new_particles;
    new_particles.reserve((unsigned long) (num_particles));

    for (int i = 0; i < num_particles; ++i)
    {
        int index = dist(gen);
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;

#ifdef DEBUG_OUTPUT
    std::cout << "----------------------------resampling--------------------------\n";
    for (int p = 0; p < particles.size(); ++p)
    {
        std::cout << "particle " << p << ": " << particles[p].x << "\t" << particles[p].y << "\t" << particles[p].theta << "\t" << particles[p].weight << std::endl;
    }
#endif
}

void ParticleFilter::write(std::string filename)
{
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i)
    {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}

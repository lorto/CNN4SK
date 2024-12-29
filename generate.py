import numpy as np
from PIL import Image
import os

# General settings
num_images = 1000 # Number of images for each execution
num_steps = 100 # Number of steps during the motion of charged particles
ellipse_points = 360 # Number of points for each ellipse
color = 255 # White color
output_dir = "event_display" # Directory to save images

# Physical and geometric parameters
theta_cherenkov = 41 * np.pi / 180 # Cherenkov angle in radians (41 degrees)
image_size = (105, 105) # Image size (the number of photomultipliers in Super-Kamiokande is about 11k)
center = (image_size[0] // 2, image_size[1] // 2) # Image center
e_noise = 5 # Standard deviation of gaussian noise in e-like events
mu_noise = 1 # Standard deviation of gaussian noise in mu-like events

# Function to emulate noise scaling
def noise_func(sigma_noise, step):
    return sigma_noise * (0.5 + 0.5 * step / num_steps)

# Function to generate (randomly) the particle's parameters
def generate_particle_parameters():
    # Generate identity, azimuthal angle, polar angle and topological type
    ID = np.random.choice(['e', 'mu']) # Electron or muon
    phi = np.random.uniform(0, 2 * np.pi) # Azimuthal angle
    theta = np.random.uniform(0, np.pi / 2) # Polar angle (0 to 90 degrees to stay in the plane)
    type = np.random.choice(['PC', 'FC']) # Partially contained topology or fully contained topology

    # Generate random distances where the Cherenkov emission starts and ends
    birth_distance_to_plane = np.random.uniform(20, 50) # The cone shouldn't be neither too big nor too small
    # If the event is FC the internal radius is set to minimum 33% of the outer
    death_distance_to_plane = 0 if type == 'PC' else np.random.uniform(0.33 * birth_distance_to_plane, birth_distance_to_plane)

    return ID, phi, theta, birth_distance_to_plane, death_distance_to_plane, type

# Function to calculate the parameters of the cone intersection with the image plane
def cone_intersection_parameters(theta, distance_to_plane):
    # Calculate the radius of the Cherenkov cone as a function of the particle's distance
    r = np.tan(theta_cherenkov) * distance_to_plane

    # Calculate the ellipse's semi-axes
    a = r / np.cos(theta) # Major semi-axis varies with the polar angle theta
    b = r # Minor semi-axis remains constant

    return a, b

# Function to calculate the ellipse points
def cone_intersection_points(phi, a, b):
    # Generate points on the ellipse
    t = np.linspace(0, 2 * np.pi, ellipse_points) # Angles to describe the ellipse (one point per degree)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Rotate the ellipse around the phi angle
    x_rot = x * np.cos(phi) - y * np.sin(phi)
    y_rot = x * np.sin(phi) + y * np.cos(phi)

    # Translate to the center of the image
    x_final = center[0] + x_rot
    y_final = center[1] + y_rot

    return x_final, y_final

# Function to add noise to the ellipse
def add_noise_to_ellipse(x_points, y_points, sigma_noise):
    # Apply noise to each point
    x_noisy, y_noisy = np.zeros(ellipse_points), np.zeros(ellipse_points)

    for t in range(ellipse_points):
        r_noise = np.abs(np.random.normal(0, sigma_noise))
        theta_noise = np.random.uniform(0, 2 * np.pi)
        x_noisy[t] = x_points[t] + r_noise * np.cos(theta_noise)
        y_noisy[t] = y_points[t] + r_noise * np.sin(theta_noise)

    return x_noisy, y_noisy

# Function to generate the image of a single event
def generate_event_image(ID, phi, theta, birth_distance_to_plane, death_distance_to_plane):
    # Create an empty (black) image
    image = np.zeros(image_size, dtype=np.uint8)

    # Calculate the parameters of the outer ellipse
    a_outer, b_outer = cone_intersection_parameters(theta, birth_distance_to_plane)

    # Calculate the points of the outer ellipse
    x_points_outer, y_points_outer = cone_intersection_points(phi, a_outer, b_outer)

    # Check if the ellipse is within the image boundaries
    if not np.all((x_points_outer >= 0) & (x_points_outer < image_size[0]) & (y_points_outer >= 0) & (y_points_outer < image_size[1])):
        return None # Ellipse outside the image, invalid event

    # Calculate the parameters of the inner ellipse
    a_inner, b_inner = cone_intersection_parameters(theta, death_distance_to_plane)

    # Calculate the difference between semi-axes to determine the intermediate steps
    a_step = (a_outer - a_inner) / num_steps
    b_step = (b_outer - b_inner) / num_steps

    # Draw the ellipses
    for step in range(num_steps + 1):
        # Calculate the new semi-axes for the intermediate ellipse
        a_current = a_outer - step * a_step
        b_current = b_outer - step * b_step

        # Calculate the points of the intermediate ellipse
        x_points, y_points = cone_intersection_points(phi, a_current, b_current)

        # Assign the right noise for the particle
        sigma_noise = e_noise if ID == 'e' else mu_noise

        # Noise increases linearly with "step" so to emulate grater multiple Coulomb scattering at the end of the path
        x_points, y_points = add_noise_to_ellipse(x_points, y_points, noise_func(sigma_noise, step))

        # Draw the intermediate ellipse
        for x, y in zip(x_points, y_points):
            if 0 <= int(x) < image_size[0] and 0 <= int(y) < image_size[1]:
                image[int(y), int(x)] = color # White for all intermediate ellipses

    return image

generated_count = 0
while generated_count < num_images:
    # Generate random parameters then generate event image
    ID, phi, theta, birth_distance_to_plane, death_distance_to_plane, type = generate_particle_parameters()
    event_image = generate_event_image(ID, phi, theta, birth_distance_to_plane, death_distance_to_plane)

    # If the image is None, it means the ellipse is out of bounds, discard the event
    if event_image is None:
        print(f"Event discarded.")
        continue # Regenerate a new event

    # Save the image as a .png file in the correct path
    folder = type + ID # Concatenate type and ID strings
    output_filename = f"{generated_count:06d}_{folder}.png" # Filename with event number, type and ID
    output_path = os.path.join(output_dir, folder, output_filename) # Output path
    directory = os.path.dirname(output_path) # Output directory

    # Create output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save image
    im = Image.fromarray(event_image)
    im.save(output_path)

    # Increment the count of generated images and print a save message
    generated_count += 1
    print(f"Event image saved as '{output_filename}'.")

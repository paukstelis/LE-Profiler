# coding=utf-8
from __future__ import absolute_import

### (Don't forget to remove me)
# This is a basic skeleton for your plugin's __init__.py. You probably want to adjust the class name of your plugin
# as well as the plugin mixins it's subclassing from. This is really just a basic skeleton to get you started,
# defining your plugin as a template plugin, settings and asset plugin. Feel free to add or remove mixins
# as necessary.
#
# Take a look at the documentation on what other plugin mixins are available.

import octoprint.plugin
import octoprint.filemanager
import octoprint.filemanager.util
import octoprint.util
import logging
import re
import os
import math
from . import G_Code_Rip as G_Code_Rip
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import root_scalar

class ProfilerPlugin(octoprint.plugin.SettingsPlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.SimpleApiPlugin,
    octoprint.plugin.TemplatePlugin,

):

    def __init__(self):
        self.plot_data = []
        self.spline = None
        self.x_coords = []
        self.z_coords = []
        self.tool_length = 0
        self.min_B = float(0)
        self.max_B = float(0)
        self.x_steps = float(0)
        self.power = float(0)
        self.start_max = False
        self.axis = 'X'
        self.side = "front"
        self.feed = 1.0
        self.arotate = 0.0
        self.segments = 0
        self.datafolder = None
        self.increment = 0.5
        self.smooth_points = 4
        self.weak_laser = 0
        self.singleB = False
        self.risky_clearance = False
        self.invert_facet = False
        #self.watched_path = self._settings.global_get_basefolder("watched")

    def initialize(self):
        self.datafolder = self.get_plugin_data_folder()
        self.gcr = G_Code_Rip.G_Code_Rip()
        self.smooth_points = int(self._settings.get(["smooth_points"]))
        self.increment  = float(self._settings.get(["increment"]))
        self.tool_length = float(self._settings.get(["tool_length"]))
        self.weak_laser = self._settings.global_get(["plugins", "latheengraver", "weakLaserValue"])
    def get_settings_defaults(self):
        return dict(
            increment=0.5,
            smooth_points=4,
            tool_length=135,
            default_segments=1,
            )
    def on_settings_save(self, data):
        octoprint.plugin.SettingsPlugin.on_settings_save(self, data)
        self.initialize()

    def get_assets(self):
        # Define your plugin's asset files to automatically include in the
        # core UI here.
        return {
            "js": ["js/Profiler.js", "js/plotly-latest.min.js"],
            "css": ["css/Profiler.css"],
        }
    
    def creategraph(self, filepath):
        folder = self._settings.getBaseFolder("uploads")
        filename = f"{folder}/{filepath}"
        
        datapoints = []
        with open(filename,"r") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line == ";X":
                    self.axis = 'X'
                if stripped_line == ";Z":
                    self.axis = 'Z'
                if not stripped_line.startswith(";"):
                    # Split the line by comma and convert to floats
                    try:
                        datapoints.append([float(x) for x in stripped_line.split(",")])
                    except ValueError:
                        pass
        self._logger.info(datapoints)
        #sort, must be increasing
        if self.axis == 'Z':
            datapoints = sorted(datapoints, key=lambda x: x[1])
            min = datapoints[0][1] #smallest X value, should be 0
            max = datapoints[-1][1] #largest X value
        if self.axis == 'X':
            datapoints = sorted(datapoints, key=lambda x: x[0])
            min = datapoints[0][0] #smallest X value, should be 0
            max = datapoints[-1][0] #largest X value

        self._logger.info(datapoints)
        self._logger.debug(self.axis)

        generated_data = []

        if self.axis == 'Z':
            z_profile, x_profile, a_profile = zip(*datapoints)
        else:
            x_profile, z_profile, a_profile = zip(*datapoints)

        self.spline = CubicSpline(x_profile, z_profile)

        increment = self.increment
        i = min
        while i <= max:
            z_val = self.spline(i)
            z_val = float(z_val)
            z_val = f"{z_val:.3f}"
            x_val = f"{i:.3f}"
            generated_data.append([x_val,z_val])
            i = i+increment
        self._logger.debug(generated_data)

        #send generated_data to plotly at the front end
        data = dict(type="graph", probe=generated_data, axis=self.axis)
        self._plugin_manager.send_plugin_message('Profiler', data)

    def create_spline(self):
        self.x_coords = []
        self.z_coords = []

        if self.axis == "X":
            for each in self.plot_data:
                self.x_coords.append(float(each["x"]))
                self.z_coords.append(float(each["z"]))
        if self.axis == "Z":
            for each in self.plot_data:
                self.z_coords.append(float(each["x"]))
                self.x_coords.append(float(each["z"]))

        self.spline = CubicSpline(self.x_coords, self.z_coords)

    def calc_coords(self, coord):
       
        closest = min(self.x_coords, key=lambda x: abs(x - coord))
        closest_idx = self.x_coords.index(closest)
        half_window = self.smooth_points // 2
        start_idx = max(0, closest_idx - half_window)
        end_idx = min(len(self.x_coords), closest_idx + half_window + 1)
        near = self.x_coords[start_idx:end_idx]
        slopes = [self.spline.derivative()(x) for x in near]
        #include the calculated value in the average
        slopes.append(self.spline.derivative()(coord))
        slope = sum(slopes) / len(slopes)
        z_value = self.spline(coord)

        #normal angle calculation
        if self.axis == "X":
            normal = math.atan2(slope, 1)
        if self.axis == "Z" and self.side == "back":
            normal = -math.atan2(slope,1) + math.pi/2
        if self.axis == "Z" and self.side == "front":
            normal = -math.atan2(slope,1) - math.pi/2
        
        b_angle = math.degrees(normal)
        
        #adjust normal angle if beyond limits
        if b_angle > 0 and b_angle > self.max_B:
            b_angle = self.max_B
        if b_angle < 0 and b_angle < self.min_B:
            b_angle = self.min_B
        #recalculate normal in case it is outside B range
        normal = math.radians(b_angle)
        
        #self._logger.info(f"Normal angle: {normal}, slope: {slope},  B angle: {b_angle} x={coord}, z={z_value}")
        
        if self.axis == "X":
            normal = normal + math.pi / 2 
            x_center = coord + ((self.tool_length) * math.cos(normal))
            z_center = z_value + ((self.tool_length) * math.sin(normal))
            return_coord = {"X": x_center, "Z": z_center-self.tool_length, "B": b_angle}
        #may need to have this specific for front and back cases
        if self.axis == "Z":
            normdir  = 1
            if self.side == "front":
                normdir = -1
            x_center = coord + ((self.tool_length) * math.cos(normal*normdir))
            z_center = z_value + ((self.tool_length) * math.sin( normdir*normal))
            return_coord = {"X": z_center-self.tool_length, "Z": x_center, "B": b_angle}
        return return_coord
    
    def generate_laser_job(self):
        command_list = []
        pass_list = []
        profile_points = []
        command_list.append(f"(LatheEngraver Laser job)")
        command_list.append(f"(Min and Max values: {self.vMin}, {self.vMax} )")
        command_list.appened(f"(Tool length: {self.tool_length})")
        command_list.append(f"(Segments: {self.segments}, A rotation: {self.arotate})")
        command_list.append(f"(B angle range: {self.min_B} to {self.max_B})")
        #truncate profile beween vMin and vMax
        for each in self.x_coords:
            if each < self.vMin:
                continue
            if each > self.vMax:
                continue
            profile_points.append(each)
            #TODO: reverse profile points if it is a Z scan
        #reverse the profile for Z axis
        if self.axis == "Z":
            profile_points.reverse()
        #A axis rotation per segment- this is very simplistic. Maybe calculate total distance and fraction of that total distace per move?
        seg_rot = self.arotate/(len(profile_points)-1)
        self._logger.info(f"Segment rotation: {seg_rot}")
        A_rot = 360/self.segments

        #Preamble stuff here
        command_list.append("G21")
        command_list.append("G90")
        
        #for our safe position(s)
        sign, safe = self.safe_retract()

        #move to start
        start = self.calc_coords(profile_points[0])
        command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
        move_1 = f"G0 X{start['X']:0.4f}"
        move_2 = f"G0 Z{start['Z']:0.4f} B{start['B']:0.4f}"
        if self.axis == "X":
            command_list.append(move_1)
            command_list.append(move_2)
        else:
            command_list.append(move_2)
            command_list.append(move_1)

        command_list.append(f"G0 X{start['X']:0.4f} Z{start['Z']:0.4f} A0 B{start['B']:0.4f}")
        if self.test:
            command_list.append(f"M4 S{self.weak_laser}")
        else:
            command_list.append(f"M4 S{self.power}")
        
        #this is to handle A rotations
        i = -1
        for each in profile_points:
            i+=1 
            coord = self.calc_coords(each)
            pass_list.append(f"G93 G90 G1 X{coord['X']:0.3f} Z{coord['Z']:0.3f} A{seg_rot*i:0.3f} B{coord['B']:0.3f} F{self.feed}")
        #make sure we move back to last A position before starting reverse pass
        pass_list.append(f"G0 A{seg_rot*i:0.3f}")    
        
        i = 1
        while i <= self.segments:
            command_list.append(f"(Starting segment {i} of {self.segments})")
            command_list.extend(pass_list)
            pass_list = pass_list[::-1]
            if self.test and i == 1:
                command_list.append("G4 P2")
                command_list.append("(test pass)")
                command_list.extend(pass_list)
                command_list.append("G4 P2")
                command_list.append("M0")
                command_list.append(f"M4 S{self.power}")
                pass_list = pass_list[::-1]
                command_list.extend(pass_list)
                pass_list = pass_list[::-1]
            #rotate
            command_list.append("G0 A0") #return A to 0 first
            command_list.append(f"G0 A{A_rot:0.3f}")
            command_list.append("G92 A0")
            i += 1
        command_list.append("M5")
        command_list.append("M30")

        output_name = self.name.removesuffix(".txt")
        output_name = f"Laser_S{self.segments}_P{self.power}_"+output_name+".gcode"
        path_on_disk = "{}/{}".format(self._settings.getBaseFolder("watched"), output_name)

        with open(path_on_disk,"w") as newfile:
            for line in command_list:
                newfile.write(f"\n{line}")

    def cut_depth_value(self, coord, depth):
        trans_x = coord["X"] + depth*math.sin(math.radians(-coord["B"]))
        trans_z = coord["Z"] + depth*math.cos(math.radians(-coord["B"]))
        return trans_x, trans_z
    
    def lead_calc(self, type, nominal_depth, step, inc):
        depth = nominal_depth
        self._logger.info(f"type={type}, nd={nominal_depth}, step={step}, inc={inc}")
        if type == "in":
            if self.axis == "X":
                depth = nominal_depth + inc*step

            if self.axis == "Z":
                if self.side == "front":
                    depth = nominal_depth + inc*step
                if self.side == "back":
                    depth= nominal_depth - inc*step

        else:  #out 
            if self.axis == "X":
                depth = nominal_depth + inc*step

            if self.axis == "Z":
                if self.side == "front":
                    depth = nominal_depth + inc*step
                if self.side == "back":
                    depth= nominal_depth - inc*step
                
        return depth
    def x_to_arc(self, profile_points, distance, start=True):
        #returns the X coordinate in our profile point that will give the arc of the length, distance
        if start:
            x_ref = profile_points[0]
            bracket= (x_ref, profile_points[-1])
        else:
            x_ref =  profile_points[-1]
            bracket= (x_ref, profile_points[0])

        def arc_length(x_target):
            integral, _ = quad(lambda x: (1 + self.spline.derivative()(x) ** 2) ** 0.5, x_ref, x_target,limit=500)
            return integral
        def root_func(x):
            return arc_length(x) - distance
        
        solution = root_scalar(root_func, bracket=bracket, method='brentq')
        if solution.converged:
            x_raw = solution.root
            closest_x = min(profile_points, key=lambda x: abs(x - x_raw))
            self._logger.info(f"converged solution: {solution.root}, closest: {closest_x}")
            return closest_x
        else:
            raise ValueError("Failed to find X coordinate for the given arc length.")

    def calc_feedrate(self, coord1, coord2):
        x1, z1 = coord1["X"], coord1["Z"]
        x2, z2 = coord2["X"], coord2["Z"]
        distance = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

        if self.feed > 0:
            time_min = (distance / self.feed)
            return (1 / time_min)
        else:
            return 0.0
        
    def sagitta_distance(self, theta, radius):

        max_angle = 2 * math.pi / self.segments
        # Clamp theta to [0, max_angle]
        if theta < 0:
            theta = 0
        elif theta > max_angle:
            theta = max_angle

        # Circle point at angle theta
        cx = radius * math.cos(theta)
        cy = radius * math.sin(theta)

        # Chord endpoints
        p1x = radius  # at angle 0
        p1y = 0

        p2x = radius * math.cos(max_angle)
        p2y = radius * math.sin(max_angle)

        # Interpolation factor t
        t = theta / max_angle

        # Interpolated point on chord
        chord_x = (1 - t) * p1x + t * p2x
        chord_y = (1 - t) * p1y + t * p2y

        # Distance from circle point to chord point
        dx = cx - chord_x
        dy = cy - chord_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        return distance
     
    def generate_facet_job(self):
        self._logger.info("Starting Facet job")
        command_list = []
        profile_points = []
        command_list.append(f"(LatheEngraver Facet job)")
        command_list.append(f"(Min and Max values: {self.vMin}, {self.vMax} )")
        command_list.append(f"(Tool length: {self.tool_length})")
        command_list.append(f"(Segments: {self.segments}, A rotation: {self.arotate})")
        command_list.append(f"(B angle range: {self.min_B} to {self.max_B})")
        step_over = self.step_over * self.cutter_diam

        #truncate profile beween vMin and vMax
        for each in self.x_coords:
            if each < self.vMin:
                continue
            if each > self.vMax:
                continue
            profile_points.append(each)

        #Preamble stuff here
        command_list.append("G21")
        command_list.append("G90")

        reference_radius = self.diam / 2
        max_radius = 0
        for x in profile_points:
            z_val = self.spline(x)
            # Calculate the radius at this point
            radius = reference_radius + (z_val - self.referenceZ)
            if radius > max_radius:
                max_radius = radius
        command_list.append(f"(Max radius: {max_radius})")
        #This is maximum negative Z at the largest radius of the piece
        max_z = max_radius*(1 - math.cos(math.pi/self.segments))
        command_list.append(f"(Max Z: {max_z})")
        #may use a scale factor here to modify max_z by some amount to avoid perfectly flat cuts
        max_z = max_z * self.depth_mod
        command_list.append(f"(Max Z with scaling: {max_z})")
        if self.depth and max_z > self.depth:
            max_z = self.depth
            command_list.append(f"(Max Z limited to {self.depth})")
        pass_info = divmod(max_z, self.step_down)
        passes = pass_info[0] #quotient
        last_pass_depth = pass_info[1] #remainder
        if last_pass_depth:
            total_passes = int(passes + 1)
        else:
            total_passes = int(passes)

        # Calculate the arc angle covered by half the tool diameter at the max radius
        tool_radius = self.cutter_diam / 2
        tool_angle_rad = tool_radius / max_radius  # radians
        tool_angle_deg = math.degrees(tool_angle_rad)

        facet_angle = 360 / self.segments
        delta_theta = (tool_radius*self.step_over) / max_radius  # radians
        delta_degrees = math.degrees(delta_theta)
        start_a = delta_degrees
        end_a = facet_angle - delta_degrees
        num_a_steps = int(math.ceil(facet_angle) / delta_degrees)
        self._logger.info(f"Depth passes: {total_passes}, Facet angle: {facet_angle}, Delta degrees: {delta_degrees}, Num A steps: {num_a_steps}")

        #A axis rotation per move this is very simplistic. Maybe calculate total distance and fraction of that total distace per move?
        seg_rot = self.arotate/(len(profile_points)-1)
        self._logger.info(f"Segment rotation: {seg_rot}")
        A_rot = 360/self.segments
        #for our safe position(s)
        sign, safe = self.safe_retract()
        #gcode generation
        #move to start
        start = self.calc_coords(profile_points[0])
        end = self.calc_coords(profile_points[-1])
        safe_x_start, safe_z_start = self.cut_depth_value(start, 5)
        safe_x_end, safe_z_end = self.cut_depth_value(end, 5)
        safe_position_start = f"G0 X{safe_x_start:0.4f} Z{safe_z_start:0.4f} B{start['B']:0.4f}"
        safe_position_end = f"G0 X{safe_x_end:0.4f} Z{safe_z_end:0.4f} B{end['B']:0.4f}"

        #command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
        #command_list.append(f"G0 B{start['B']:0.4f}")
        #command_list.append(f"G0 X{safe_x_start:0.4f}")
        #command_list.append(f"G0 Z{safe_z_start:0.4f}")

        facet_list = []
        current_a = 0
        a_direction = 1
        max_zmod = 0
        current_a = start_a
        a_measure = start_a
        ease_down = True

        for a_step in range(num_a_steps):
            section_done = False
            max_zmod = 0
            previous_depth = 0    
            if a_measure > end_a:
                continue
            facet_list.append(f"(Facet A angle step {a_step+1} of {num_a_steps})")
            facet_list.append(f"(Current A angle: {current_a:.3f})")
            facet_list.append(f"(A measure: {a_measure:.3f})")
            facet_list.append(f"G0 A{current_a:.3f}")

            for depth in range(1,total_passes+1):
                previous_coord = None
                nominal_depth = depth * self.step_down
                               
                if depth > 1:
                    if nominal_depth >= max_zmod and previous_depth >= max_zmod:
                        self._logger.info(f"Max Z mod reached: {max_zmod:.2f}, stopping section")
                        section_done = True
                    if nominal_depth >= max_zmod:
                        nominal_depth = max_zmod
                
                self._logger.info(f"Nominal depth: {nominal_depth}")
                previous_depth = nominal_depth
                i = 0
                if not section_done:
                    x_iter = profile_points if a_direction == 1 else reversed(profile_points)
                    facet_list.append(f"(Facet depth pass {depth} with nominal depth: {nominal_depth})")
                    for x in x_iter:
                        coord = self.calc_coords(x)
                        retract_x, retract_z = self.cut_depth_value(coord, 5)
                        if previous_coord:
                            feed = self.calc_feedrate(previous_coord, coord)
                        else:
                            feed = self.feed
                        previous_coord = coord
                        

                        #get radius at our current X position
                        current_radius = reference_radius + (self.spline(x) - self.referenceZ)
                        #self._logger.info(f"Current radius: {current_radius}")
                        #just use z_mod values at max_radius for inverts
                        if self.invert_facet:
                            current_radius = max_radius
                        #get adjusted values for Z
                        z_mod = self.sagitta_distance(math.radians(a_measure),current_radius)
                        z_mod = z_mod * self.depth_mod  # Apply depth modifier

                        if self.invert_facet:
                            z_mod = max_z - z_mod
    
                        if z_mod > max_zmod:
                            max_zmod = z_mod
                            #self._logger.info(f"New max Z mod: {max_zmod:.2f}")

                        if z_mod > nominal_depth:
                            z_mod = nominal_depth
                            if depth == 1 and ease_down:
                                fract = nominal_depth/20
                                z_mod = fract*(i+1)
                                facet_list.append(f"(Ease down step with z_mod: {z_mod:.2f})")
                                if z_mod > nominal_depth:
                                    facet_list.append(f"(Ease down done)")
                                    z_mod = nominal_depth
                                    ease_down = False

                        trans_x, trans_z = self.cut_depth_value(coord, -z_mod)  # Adjust depth
                        #handle A rotation parameter
                        a_move = current_a
                        if seg_rot:
                            a_move = current_a + (seg_rot * i * a_direction)
                        
                        facet_list.append(f"G93 G1 X{trans_x:.3f} Z{trans_z:.3f} A{a_move:.3f} B{coord['B']:.3f} F{feed:.1f}")
                        i += 1
                    a_direction *= -1        
                else:
                    #if we are done with this pass, retract from current position
                    facet_list.append(f"(Section done, retracting to safe position)")
                    facet_list.append(f"G0 X{retract_x:.3f} Z{retract_z:.3f} B{coord['B']:.3f}")
            #if we haven't hit ease down on the first segment, ignore
            ease_down = False
            if seg_rot:
                current_a = a_move
            current_a += delta_degrees
            a_measure += delta_degrees
            facet_list.append(f"G0 X{retract_x:.3f} Z{retract_z:.3f} B{coord['B']:.3f}")
                
        for j in range(0, self.segments):
            command_list.append(f"(Starting facet {j+1} of {self.segments})")
            command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
            command_list.append(f"G0 B{start['B']:0.4f}")
            command_list.append(f"G0 X{safe_x_start:0.4f}")
            command_list.append(f"G0 Z{safe_z_start:0.4f}")
            command_list.extend(facet_list)
            command_list.append(f"G0 A{facet_angle:0.3f}")
            command_list.append("G92 A0")
       
        command_list.append("M5")
        command_list.append("M30")
        output_name = self.name.removesuffix(".txt")
        invert = ""
        if self.invert_facet:
            invert = "inv"
        output_name = f"Facet_S{self.segments}_Arot{self.arotate}_SD{self.step_down}_{invert}_"+output_name+".gcode"
        path_on_disk = "{}/{}".format(self._settings.getBaseFolder("watched"), output_name)
        with open(path_on_disk,"w") as newfile:
            for line in command_list:
                newfile.write(f"\n{line}")

    def generate_flute_job(self):
        self._logger.info("Starting Flute job")
        command_list = []
        profile_points = []
        command_list.append(f"(LatheEngraver Flute job)")
        command_list.append(f"(Min and Max values: {self.vMin}, {self.vMax} )")
        command_list.append(f"(Tool length: {self.tool_length})")
        command_list.append(f"(Segments: {self.segments}, A rotation: {self.arotate})")
        command_list.append(f"(Depth: {self.depth}, Step down: {self.step_down})")
        command_list.append(f"(Lead-in: {self.leadin}, Lead-out: {self.leadout})")
        command_list.append(f"(Feed: {self.feed})")
        command_list.append(f"(B angle range: {self.min_B} to {self.max_B})")

        #truncate profile beween vMin and vMax
        for each in self.x_coords:
            if each < self.vMin:
                continue
            if each > self.vMax:
                continue
            profile_points.append(each)
        
        #reverse the profile for Z axis
        if self.axis == "Z":
            profile_points.reverse()

        if self.axis == "X":
            z_at_min = self.spline(self.vMin)
            z_at_max = self.spline(self.vMax)
            if z_at_max < z_at_min:
                profile_points.reverse()

        #A axis rotation per segment- this is very simplistic. Maybe calculate total distance and fraction of that total distace per move?
        seg_rot = self.arotate/(len(profile_points)-1)
        self._logger.info(f"Segment rotation: {seg_rot}")
        A_rot = 360/self.segments
        #for our safe position(s)
        sign, safe = self.safe_retract()

        #Preamble stuff here
        command_list.append("G21")
        command_list.append("G90")
        #move to start
        start = self.calc_coords(profile_points[0])
        trans_x, trans_z = self.cut_depth_value(start, 5)
        self._logger.info(f"Start coords: X{start['X']}, Z{start['Z']}. Modified X{trans_x}, Z{trans_z}")
        safe_position = f"G0 X{trans_x:0.4f} Z{trans_z:0.4f} B{start['B']:0.4f}"
        
        command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
        move_1 = f"G0 X{trans_x:0.4f}"
        move_2 = f"G0 Z{trans_z:0.4f} B{start['B']:0.4f}"
        if self.axis == "X":
            command_list.append(move_1)
            command_list.append(move_2)
        else:
            command_list.append(move_2)
            command_list.append(move_1)
        command_list.append(safe_position)
        command_list.append(f"M3 S24000")

        #calculate how many depth passes we need
        pass_info = divmod(self.depth, self.step_down)
        passes = pass_info[0] #quotient
        last_pass_depth = pass_info[1] #remainder
        if last_pass_depth:
            total_passes = passes + 1
        else:
            total_passes = passes

        #calculate lead-in/out increments
        if self.leadin or self.leadout:
            try:
                #get profile x value from start/end that closest to the desired leadin/leadout:
                lead_in_x = self.x_to_arc(profile_points, self.leadin, start=True)
                lead_out_x = self.x_to_arc(profile_points, -self.leadout, start=False)

                if lead_out_x < lead_in_x:
                    #lead in's and outs  overlap, abort,throw some error or something
                    self._plugin_manager.send_plugin_message("latheengraver", dict(type="simple_notify",
                                                                    title="Lead-in/Lead-out error",
                                                                    text="Lead-in and lead-out overlap, please adjust values",
                                                                    hide=True,
                                                                    delay=10000,
                                                                    notify_type="error"))
                    return
            except:
                self._logger.info("Yeah leadin/out failed")
            total_in_step = int((lead_in_x - profile_points[0])/self.increment)
            total_out_step = int((profile_points[-1] - lead_out_x)/self.increment)
            #DOC, need to redo how this works.
            in_inc = self.step_down/(total_in_step)
            out_inc = self.step_down/(total_out_step)
            self._logger.info(f"steps lead-in: {total_in_step}, lead-out {total_out_step}")
            self._logger.info(f"increment for lead-in: {in_inc}, lead-out {out_inc}") 

        current_pass = 1
        while current_pass <= total_passes:
            depth = 0
            pass_list = []
            previous_coord = None
    
            #calculate depth on this pass
            nominal_depth = current_pass*self.step_down*-1
            if current_pass == total_passes and last_pass_depth:
                nominal_depth = self.depth*-1
            
            pass_list.append(f"(Cut depth: {nominal_depth})")
            i = -1 #handles A rotations better
            for each in profile_points:
                i+=1
                steps_from_end = len(profile_points) - 1 - i
                #Modify Z amounts for lead-in/out, logic may have to change for Z-axis cuts
                if self.leadin and i < total_in_step:
                    depth = self.lead_calc("in", nominal_depth, total_in_step - i, in_inc)
                if self.leadout and steps_from_end < total_out_step:
                    depth = self.lead_calc("out", nominal_depth, total_out_step - steps_from_end, out_inc)
                self._logger.info(f"Depth: {depth}")
                coord = self.calc_coords(each) #these just follow profile, have to add cut depth
                #get adjusted values
                if previous_coord:
                    feed = self.calc_feedrate(previous_coord, coord)
                else:
                    feed = self.feed
                previous_coord = coord               
                trans_x, trans_z = self.cut_depth_value(coord, depth)
                pass_list.append(f"G93 G90 G1 X{trans_x:0.3f} Z{trans_z:0.3f} A{seg_rot*i:0.3f} B{coord['B']:0.3f} F{feed:0.1f}")
                depth = nominal_depth
            #Go to safe position from latest coord, this needs to retract ALONG current B
            trans_x, trans_z = self.cut_depth_value(coord, 5)

            pass_list.append("(Pass done, move to safe position)")
            pass_list.append(f"G0 X{trans_x:0.3f} Z{trans_z:0.3f} B{coord['B']:0.3f}")
            if not self.risky_clearance:
                pass_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
            start_x, start_z = self.cut_depth_value(start, 5)
            pass_list.append(f"G90 G0 {self.axis}{start_x:0.3f}")
            #make sure we move back to last A position before starting next pass
            pass_list.append(f"G0 A{seg_rot*i:0.3f}")
            #move to clear position
            
            j = 1
            while j <= self.segments:
                command_list.append(f"(Starting segment {j} of {self.segments})")
                command_list.extend(pass_list)
                command_list.append("G0 A0") #return A to 0 first
                command_list.append(f"G0 A{A_rot:0.3f}")
                command_list.append("G92 A0")
                #move to start safe position for next pass:
                command_list.append(safe_position)  
                j += 1
            current_pass += 1
        command_list.append("M5")
        command_list.append("M30")
        output_name = self.name.removesuffix(".txt")
        output_name = f"Flute_S{self.segments}_Arot{self.arotate}_D{self.depth}_"+output_name+".gcode"
        path_on_disk = "{}/{}".format(self._settings.getBaseFolder("watched"), output_name)

        with open(path_on_disk,"w") as newfile:
            for line in command_list:
                newfile.write(f"\n{line}")

    def arc_length(self, x):
        spline_derivative = self.spline.derivative()
        return (1 + spline_derivative(x) ** 2) ** 0.5
    
    def get_arc(self, x1, x2):
        profile_dist, _ = quad(self.arc_length, x1, x2, limit=500)
        return profile_dist

    def generate_wrap_job(self):
        #create profile from diameter reference
        profile_points = []
        
        #truncate profile beween vMin and vMax
        for each in self.x_coords:
            if each < self.vMin:
                continue
            if each > self.vMax:
                continue
            profile_points.append(each)

        #gcr = G_Code_Rip.G_Code_Rip()
        basefolder = self._settings.getBaseFolder("uploads")
        self.gcr.Read_G_Code(f"{basefolder}/{self.selected_file}", XYarc2line=True, units="mm")
        #profile name
        self._logger.info(self.gcr.g_code_data)
        #make the first move a safe X,Z move

        profile_name = self.name.removesuffix(".txt")
        gcode_name = os.path.basename(self.selected_file).removesuffix(".gcode")
        output_name = f"Wrap_{profile_name}_{gcode_name}.gcode"
        #calculate scalefactor
        profile_dist = self.get_arc(self.vMin, self.vMax)
        sf = profile_dist/self.width
        new_width = self.width*sf
        self._logger.info(f"Profile distance: {profile_dist}, Scale factors: {sf}, New width:{new_width}")
        #now get the X position that will correspond to that length along thearc
        target_x = self.x_to_arc(profile_points, new_width, start=True)
        
        a = target_x-self.vMin

        xtoscale = (a)/self.width
        self._logger.info(f"Target X-value={target_x}, xtoscale={xtoscale}")
        #have to go back and handle z cases too
        temp,minx,maxx,miny,maxy,minz,maxz  = self.gcr.scale_rotate_code(self.gcr.g_code_data,
                                                                    [xtoscale,sf,1,1],
                                                                    0,
                                                                    split_moves=True,
                                                                    min_seg_length=self.steps)
        #self._logger.info(temp)
        midx = (minx+maxx)/2
        midy = (miny+maxy)/2
        #self._logger.info(self.plot_data)
        self._logger.info(f"midx: {midx}")
        #calculate offset, just x for now
        #xoffset = (self.vMax+self.vMin)/2 + self.vMin
        xoffset = abs(minx) + self.vMin
        self._logger.info(f"X offset: {xoffset}")
        temp = self.gcr.scale_translate(temp,translate=[-xoffset,0,0.0])
        
        
        temp = self.gcr.profile_conform(temp,
                                        self.spline,
                                        profile_points,
                                        self.min_B,
                                        self.max_B,
                                        self.tool_length,
                                        self.diam/2,
                                        self.radius_adjust,
                                        self.referenceZ,
                                        self.singleB,
                                        self.smoothing)
        self._logger.info(temp)
        #get first X and Z moves that are not complex
        first_x = None
        first_z = None
        for line in temp:
            if line[0] == 0 or line[0] == 1:
                if not isinstance(line[1][3], complex):
                    first_z = line[1][2]
                if not isinstance(line[1][1], complex):
                    first_x = line[1][0]
                    break


        path_on_disk = "{}/{}".format(self._settings.getBaseFolder("watched"), output_name)
        if self.segments > 1:
            arots = 360/self.segments
        repeats = self.segments
        sign, safe = self.safe_retract()
        with open(path_on_disk,"w") as newfile:
            newfile.write(f"(LatheEngraver G-code Wrapping)\n")
            newfile.write(f"(Ref. Diam: {self.diam}, Projection: {self.radius_adjust}, G-code width: {self.width})\n")
            newfile.write(f"(Profile distance: {profile_dist:0.2f}, X-scale factor: {xtoscale}, A-scale factor: {sf})\n")
            newfile.write("(Safe moves added)\n")

            newfile.write(f"G90 G21\n")
            newfile.write(f"G0 {safe}{sign}{10+self.clearance:0.4f}\n")
            newfile.write(f"G0 X{first_x:0.2f}\n")
            #single case
            while repeats:
                for line in self.gcr.generategcode(temp,Rstock=self.diam/2,no_variables=True,Wrap="SPECIAL",FSCALE="None"):
                    if repeats > 1 and line.startswith("M30"):
                        continue
                    else:
                        newfile.write(f"\n{line}")
                repeats -= 1
                if repeats:
                    newfile.write("\nG0 A0")
                    newfile.write(f"\nG0 A{arots:0.4f}")
                    newfile.write("\nG92 A0")
            
    def safe_retract(self):
        sign = ""
        safe = None
        if self.axis == "X":
            safe = "Z"
        if self.axis == "Z":
            if self.side == "back":
                sign = "-"
            safe = "X"
        return sign, safe
    
    def is_api_protected(self):
        return True
    
    def get_api_commands(self):
        return dict(
            write_job=[],
            go_to_position=[],
            creategraph=[],
            get_arc_length=[],
        )
    
    def on_api_command(self, command, data):
        
        if command == "creategraph":
            filePath = data["filepath"]
            self.creategraph(filePath)
            return
        
        if command == "write_job":
            self.plot_data = data["plot_data"]
            self.mode = data["mode"]
            self.tool_length = float(data["tool_length"])
            self.max_B = float(data["max_B"])
            self.min_B = float(data["min_B"])
            self.clearance = float(data["clear"])
            self.side = data["side"]
            self.name = data["name"]
            self.arotate = float(data["arotate"])
            self.segments = int(data["segments"])
            self.steps = float(data["steps"])
            self.smoothing = int(data["smoothing"])
            if self.segments == 0:
                self.segments = 1
            self.vMax = float(data["vMax"])
            self.vMin = float(data["vMin"])
            self.feed = int(data["feed"])
            self.risky_clearance = bool(data["risky"])
            #must sort data first
            for each in self.plot_data:
                for k, v in each.items():
                    each[k] = float(v)

            if self.axis == "X":
                self.plot_data = sorted(self.plot_data, key=lambda x: x["x"])
            if self.axis == "Z":
                self.plot_data = sorted(self.plot_data, key=lambda x: x["z"])
                #self._logger.info(self.plot_data)
            self.create_spline()

            if self.mode == "laser":
                self.test = bool(data["test"])
                self.power = int(data["power"])

                #self.start_max = bool(data["start"])
                self.generate_laser_job()

            if self.mode == "flute":
                self.depth = float(data["depth"])
                self.step_down = float(data["step_down"])
                self.leadin = float(data["leadin"])
                self.leadout = float(data["leadout"])
                self.generate_flute_job()
            
            if self.mode == "wrap":
                self.referenceZ = float(data["refZ"])
                self.width = float(data["width"]) #width in X as reported by bgs
                self.selected_file = data["filename"]["path"]
                self.diam = float(data["diam"])
                self.radius_adjust = bool(data["radius_adjust"])
                self.singleB = bool(data["singleB"])
                self.generate_wrap_job()
            
            if self.mode == "facet":
                self.referenceZ = float(data["refZ"])
                self.diam = float(data["diam"])
                self.step_down = float(data["step_down"])
                self.cutter_diam = float(data["tool_diam"])
                self.step_over = float(data["step_over"])
                self.invert_facet = bool(data["facet_invert"])
                self.depth_mod = float(data["depth_mod"])
                self.depth = float(data["depth"])
               
                if self.segments < 3:
                    self._plugin_manager.send_plugin_message("latheengraver", dict(type="simple_notify",
                                                                    title="Facet Error",
                                                                    text="Minimum segments for Facet is 3",
                                                                    hide=True,
                                                                    delay=10000,
                                                                    notify_type="error"))
                    return
                self.generate_facet_job()
        if command == "get_arc_length":
            self.vMax = float(data["vMax"])
            self.vMin = float(data["vMin"])
            profile_distance = self.get_arc(self.vMin, self.vMax)
            self._logger.info(f"Profile distance: {profile_distance}")
            data = dict(type="distance", pd=f"{profile_distance:0.2f}")
            self._plugin_manager.send_plugin_message('Profiler', data)
            
        if command == "go_to_position":
            self.plot_data = data["plot_data"]
            self.target = float(data["target"])
            self.clearance = float(data["clear"])
            self.tool_length = float(data["tool_length"])
            self.max_B = float(data["max_B"])
            self.min_B = float(data["min_B"])
            self.side = data["side"]
            #must sort data first
            for each in self.plot_data:
                for k, v in each.items():
                    each[k] = float(v)
            if self.axis == "X":
                self.plot_data = sorted(self.plot_data, key=lambda x: x["x"])
            if self.axis == "Z":
                self.plot_data = sorted(self.plot_data, key=lambda x: x["z"])
            self.create_spline()

            sign, safe = self.safe_retract()

            #self._logger.info(self.x_coords)
            #Move to safe position
            gcode = ["G90","G21",f"G0 {safe}{sign}{10+self.clearance:0.4f}"]
            coord = self.calc_coords(self.target)
            b_move  = (f"G90 B{coord['B']:0.4f}")
            move_1 = (f"G93 G90 G0 X{coord['X']:0.4f}")
            move_2 = (f"G93 G90 G0 Z{coord['Z']:0.4f}")
            if self.axis == "X":
                gcode.append(b_move)
                gcode.append(move_1)
                gcode.append(move_2)
            else:
                gcode.append(b_move)
                gcode.append(move_2)
                gcode.append(move_1)
            self._logger.info(gcode)
            self._printer.commands(gcode)


    def get_update_information(self):
        # Define the configuration for your plugin to use with the Software Update
        # Plugin here. See https://docs.octoprint.org/en/master/bundledplugins/softwareupdate.html
        # for details.
        return {
            "Profiler": {
                "displayName": "Profiler",
                "displayVersion": self._plugin_version,

                # version check: github repository
                "type": "github_release",
                "user": "paukstelis",
                "repo": "LE-Profiler",
                "current": self._plugin_version,

                # update method: pip
                "pip": "https://github.com/paukstelis/LE-Profiler/archive/{target_version}.zip",
            }
        }


# If you want your plugin to be registered within OctoPrint under a different name than what you defined in setup.py
# ("OctoPrint-PluginSkeleton"), you may define that here. Same goes for the other metadata derived from setup.py that
# can be overwritten via __plugin_xyz__ control properties. See the documentation for that.
__plugin_name__ = "Profiler"


# Set the Python version your plugin is compatible with below. Recommended is Python 3 only for all new plugins.
# OctoPrint 1.4.0 - 1.7.x run under both Python 3 and the end-of-life Python 2.
# OctoPrint 1.8.0 onwards only supports Python 3.
__plugin_pythoncompat__ = ">=3,<4"  # Only Python 3

def __plugin_load__():
    global __plugin_implementation__
    __plugin_implementation__ = ProfilerPlugin()

    global __plugin_hooks__
    __plugin_hooks__ = {
        "octoprint.plugin.softwareupdate.check_config": __plugin_implementation__.get_update_information
    }

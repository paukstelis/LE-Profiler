# coding=utf-8
from __future__ import absolute_import

import octoprint.plugin
import octoprint.filemanager
import octoprint.filemanager.util
import octoprint.util
import logging
import re
import os
import math
import time
from . import G_Code_Rip as G_Code_Rip

from scipy.interpolate import RectBivariateSpline

from scipy.integrate import quad
from scipy.optimize import root_scalar
import numpy as np
class ProfilerPlugin(octoprint.plugin.SettingsPlugin,
    octoprint.plugin.AssetPlugin,
    octoprint.plugin.StartupPlugin,
    octoprint.plugin.SimpleApiPlugin,
    octoprint.plugin.TemplatePlugin,

):

    def __init__(self):
        self.plot_data = []
        self.spline = None
        self.a_spline = None
        self.ind_v = []
        self.dep_v = []
        self.start_min= None
        self.start_max = None
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
        self.new_increment = self.increment
        self.smooth_points = 36
        self.weak_laser = 0
        self.singleB = False
        self.risky_clearance = False
        self.invert_facet = False
        self.do_oval = False
        self.adaptive = False
        self.feedscale = 1.0
        self.writing = False
        self.conventional = False
        self.use_m3 = False
        self.feed_correct = 2
        self.use_pchip = False
        self.splinetype = None

        #self.watched_path = self._settings.global_get_basefolder("watched")

    def initialize(self):
        self.datafolder = self.get_plugin_data_folder()
        self.gcr = G_Code_Rip.G_Code_Rip()
        self.smooth_points = int(self._settings.get(["smooth_points"]))
        self.increment  = float(self._settings.get(["increment"]))
        self.new_increment = self.increment
        #self.tool_length = float(self._settings.get(["tool_length"]))
        self.use_m3 = bool(self._settings.get(["use_m3"]))
        self._logger.info(f"Use m3 is {self.use_m3}")
        self.weak_laser = self._settings.global_get(["plugins", "latheengraver", "weakLaserValue"])
        self.pchip = bool(self._settings.get(["pchip"]))
        if self.pchip:
            from scipy.interpolate import PchipInterpolator
            self.splinetype = PchipInterpolator
            self._logger.info("Using PChip interpolation")
        else:
            from scipy.interpolate import CubicSpline
            self.splinetype = CubicSpline
            self._logger.info("Using CubicSpline interpolation")
        storage = self._file_manager._storage("local")
        if storage.folder_exists("wrap"):
            self._logger.info("wrap folder exists")
        else:
            storage.add_folder("wrap")

    def get_settings_defaults(self):
        return dict(
            increment=0.25,
            smooth_points=100,
            default_segments=1,
            use_m3=False,
            )
    
    def get_template_configs(self):
        return [
            dict(type="settings", name="Profiler", custom_bindings=False)
        ]
    
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
    
    def _parse_g0(self, line: str):
        if not line.lstrip().upper().startswith(("G0", "G00")):
            return None, None, None
        pairs = re.findall(r'([ABXYZ])\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)', line, flags=re.IGNORECASE)
        vals = {k.upper(): float(v) for k, v in pairs}
        return vals.get("X"), vals.get("Z"), vals.get("B")
    
    def creategraph(self, filepath):
        folder = self._settings.getBaseFolder("uploads")
        filename = f"{folder}/{filepath}"
        self.do_oval = False
        
        datapoints = []
        segments = []
        current_segment = []
        with open(filename,"r") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line == ";X":
                    self.axis = 'X'
                if stripped_line == ";Z":
                    self.axis = 'Z'
                if stripped_line == "NEXTSEGMENT":
                    segments.append(current_segment)
                    self.do_oval = True
                    current_segment = []
                    continue
                if not stripped_line.startswith(";"):
                    # Split the line by comma and convert to floats
                    try:
                        parts = [float(x) for x in stripped_line.split(",")]
                        # Pad to 3 elements
                        while len(parts) < 3:
                            parts.append(0.0)
                        current_segment.append(parts)
                    except ValueError:
                        pass
        if not len(segments):
            segments.append(current_segment)
        
        if len(segments) > 1:
            #zero_segment = list(segments[0])
            #duplicate zero segment at 360
            #for each in zero_segment:
            #    each[2] = 360.0
            #segments.append(zero_segment)
            self.do_oval = True

        #self._logger.info(segments)
        
        arr = np.array(segments)
        #sort, must be increasing
        if self.axis == 'Z':
            for seg in segments:
                seg.sort(key=lambda x: x[1])
            self.ind_v = [x[1] for x in segments[0]]
            self.dep_v = [x[0] for x in segments[0]]
            ind_vals = arr[0, :, 1]
            A_vals = arr[:, 0, 2]
            baseline_dep = arr[0, :, 0]
            dep_raw = arr[:, :, 0]
            dep_grid = dep_raw - baseline_dep

        if self.axis == 'X':
            for seg in segments:
                seg.sort(key=lambda x: x[0])
            self.ind_v = [x[0] for x in segments[0]]
            self.dep_v = [x[1] for x in segments[0]]
            ind_vals = arr[0, :, 0]
            A_vals = arr[:, 0, 2]
            baseline_dep = arr[0, :, 1]
            dep_raw = arr[:, :, 1]
            dep_grid = dep_raw - baseline_dep

        min = self.ind_v[0] 
        max = self.ind_v[-1]
        #hold on to our reference min and max values if 0,0 changes
        self.start_min = min
        self.start_max = max
        #hold on to original arrays for A spline creation
        self.ind_vals = ind_vals
        self.A_vals = A_vals
        self.dep_grid = dep_grid

        generated_data = []
        #spline used for graphing, all at A=0
        self.spline = self.splinetype(self.ind_v, self.dep_v)

        increment = self.increment
        i = min
        while i <= max:
            z_val = self.spline(i)
            z_val = float(z_val)
            z_val = f"{z_val:.2f}"
            x_val = f"{i:.2f}"
            generated_data.append([x_val,z_val])
            i = i+increment
        self._logger.debug(generated_data)

        #send generated_data to plotly at the front end
        data = dict(type="graph", probe=generated_data, axis=self.axis)
        self._plugin_manager.send_plugin_message('Profiler', data)

    def create_spline(self):
        self.ind_v = []
        self.dep_v = []

        if self.axis == "X":
            for each in self.plot_data:
                self.ind_v.append(float(each["x"]))
                self.dep_v.append(float(each["z"]))
        if self.axis == "Z":
            for each in self.plot_data:
                self.ind_v.append(float(each["z"]))
                self.dep_v.append(float(each["x"]))

        self.spline = self.splinetype(self.ind_v, self.dep_v)

    def create_a_spline(self):
        #do any ind_val offsets here?
        current_max = self.ind_v[-1]
        current_min = self.ind_v[0]
        
        sort_idx = np.argsort(self.ind_vals)
        self.ind_vals = self.ind_vals[sort_idx]
        self.dep_grid = self.dep_grid[:, sort_idx]

        self._logger.info(self.ind_vals)
        self._logger.info(self.dep_grid)
        A_radians = np.deg2rad(np.mod(self.A_vals, 360.0))
        if A_radians[0] == 0 and self.A_vals[-1] == 360:
            A_radians = np.append(A_radians, 2 * np.pi)
            Z_grid = np.vstack([Z_grid, Z_grid[0]])
        self.a_spline = RectBivariateSpline(A_radians, self.ind_vals, self.dep_grid, kx=3, ky=3, s=0)

    def ovality_mod(self, x, a_deg):
        a_wrapped = np.deg2rad(np.mod(a_deg, 360.0))
        return self.a_spline.ev(a_wrapped, x)
    
    def calc_coords(self, coord):
       
        closest = min(self.ind_v, key=lambda x: abs(x - coord))
        closest_idx = self.ind_v.index(closest)
        half_window = self.smooth_points // 2
        start_idx = max(0, closest_idx - half_window)
        end_idx = min(len(self.ind_v), closest_idx + half_window + 1)
        near = self.ind_v[start_idx:end_idx]
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
    
    def cut_depth_value(self, coord, depth):
        trans_x = coord["X"] + depth*math.sin(math.radians(-coord["B"]))
        trans_z = coord["Z"] + depth*math.cos(math.radians(-coord["B"]))
        return trans_x, trans_z
    
    def lead_calc(self, type, nominal_depth, step, inc):
        depth = nominal_depth
        self._logger.debug(f"type={type}, nd={nominal_depth}, step={step}, inc={inc}")
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
    
    def x_to_arc(self, profile_points, distance, start=True, raw=False):
        #returns the X coordinate in our profile point that will give the arc of the length, distance
        pp = profile_points
        
        if start:
            x_ref = pp[0]
            bracket= (x_ref, pp[-1])
        else:
            x_ref =  pp[-1]
            bracket= (x_ref, pp[0])

        def arc_length(x_target):
            integral, _ = quad(lambda x: math.sqrt(1 + self.spline.derivative()(x) ** 2), x_ref, x_target,limit=500)
            return integral
        def root_func(x):
            return arc_length(x) - distance
        
        solution = root_scalar(root_func, bracket=bracket, method='brentq')
        if solution.converged:
            x_raw = solution.root
            if raw:
                return x_raw
            closest_x = min(pp, key=lambda x: abs(x - x_raw))
            self._logger.info(f"converged solution: {solution.root}, closest: {closest_x}")
            return closest_x
        else:
            raise ValueError("Failed to find X coordinate for the given arc length.")

    def calc_feedrate(self, coord1, coord2):
        x1, z1, b1 = coord1["X"], coord1["Z"], coord1["B"]
        x2, z2, b2 = coord2["X"], coord2["Z"], coord2["B"]
        b1 = np.deg2rad(-b1)
        b2 = np.deg2rad(-b2)
        xt1 = x1 - self.tool_length*np.sin(b1)
        zt1 = z1 - self.tool_length*np.cos(b1)
        xt2 = x2 - self.tool_length*np.sin(b2)
        zt2 = z2 - self.tool_length*np.cos(b2)
        ds = np.sqrt((xt2-xt1)**2 + (zt2-zt1)**2)
        feed = self.feed/ds
        return feed

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

    def arc_length(self, x):
        spline_derivative = self.spline.derivative()
        return (1 + spline_derivative(x) ** 2) ** 0.5
    
    def get_arc(self, x1, x2):
        profile_dist, _ = quad(self.arc_length, x1, x2, limit=500)
        return profile_dist

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
    
    def resample_profile(self):
        profile_points = []
        domain_min = max(self.vMin, self.ind_v[0])
        domain_max = min(self.vMax, self.ind_v[-1])

        if domain_min > domain_max:
            self._logger.warning(f"Resample domain invalid: [{domain_min}, {domain_max}]")
        else:
            # np.arange may drop the last value; ensure domain_max is included
            steps = int(math.floor((domain_max - domain_min) / self.new_increment))
            for k in range(steps + 1):
                x = domain_min + k * self.new_increment
                profile_points.append(x)
            if not math.isclose(profile_points[-1], domain_max, abs_tol=1e-9):
                profile_points.append(domain_max)
        return profile_points
    
    def generate_laser_job(self):
        data = dict(title="Writing Gcode...", text="Laser job is writing.", delay=60000, type="info")
        self.send_le_message(data)
        command_list = []
        pass_list = []
        feed = self.feed
        if self.use_m3:
            fire = "M3"
        else:
            fire = "M4"
        command_list.append(f"(LatheEngraver Laser job)")
        command_list.append(f"(Min and Max values: {self.vMin}, {self.vMax} )")
        command_list.append(f"(Tool length: {self.tool_length})")
        command_list.append(f"(Segments: {self.segments}, A rotation: {self.arotate})")
        command_list.append(f"(B angle range: {self.min_B} to {self.max_B})")
        command_list.append(f"(Fixed axis increment: {self.new_increment})")
        command_list.append(f"(B-angle smoothing points: {self.smooth_points})")
        profile_points = self.resample_profile()

        self._logger.info(profile_points)
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
        command_list.append("G90")
        command_list.append("G94")
        #command_list.append(f"F{self.feed}")
        command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
        move_1 = f"G0 X{start['X']:0.4f}"
        move_2 = f"G0 Z{start['Z']:0.4f}"
        b_move = f"G0 B{start['B']:0.4f}"
        command_list.append(b_move)
        if self.axis == "X":
            command_list.append(move_1)
            command_list.append(move_2)
        else:
            command_list.append(move_2)
            command_list.append(move_1)

        command_list.append(f"G0 X{start['X']:0.4f} Z{start['Z']:0.4f} A0 B{start['B']:0.4f}")
        if self.test:
            command_list.append(f"{fire} S{self.weak_laser}")
        else:
            command_list.append(f"{fire} S{self.power}")
        
        #command_list.append(f"F{self.feed}")
        #this is to handle A rotations
        i = -1
        previous_coord = None
        previous_feed = None
        for each in profile_points:
            
            i+=1 
            coord = self.calc_coords(each)

            if self.feed_correct == 1:
                if i >= 1:
                    linear_distance = abs(self.get_arc(profile_points[i-1],profile_points[i]))
                    feed = self.feed*(self.new_increment/linear_distance)
                    self._logger.debug(f"linear distance: {linear_distance} at profile_point: {profile_points[i]} scaled to feed {feed}")

            if self.feed_correct == 2:
                if previous_coord:
                    feed = self.calc_feedrate(previous_coord, coord)
                else:
                    feed = self.feed

            if previous_feed:
                ratio = 1.25
                feed = max(previous_feed/ratio, min(feed, previous_feed*ratio))

            pass_list.append(f"G93 G1 X{coord['X']:0.3f} Z{coord['Z']:0.3f} A{seg_rot*i:0.3f} B{coord['B']:0.3f} F{feed:0.1f}")

            previous_coord = coord
            previous_feed = feed
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
                command_list.append(f"{fire} S{self.power}")
                pass_list = pass_list[::-1]
                command_list.extend(pass_list)
                pass_list = pass_list[::-1]
            #rotate
            command_list.append("G0 A0") #return A to 0 first
            command_list.append(f"G0 A{A_rot:0.3f}")
            command_list.append("G92 A0")
            i += 1
        command_list.append("G94")
        command_list.append("M5")
        command_list.append("M30")

        output_name = self.name.removesuffix(".txt")
        output_name = f"Laser_S{self.segments}_P{self.power}_"+output_name+".gcode"
        path_on_disk = "{}/{}".format(self._settings.getBaseFolder("watched"), output_name)

        with open(path_on_disk,"w") as newfile:
            for line in command_list:
                newfile.write(f"\n{line}")

        self.send_le_clear()
        self._plugin_manager.send_plugin_message('latheengraver',  dict(type='filerefresh'))

    def generate_facet_job(self):
        self._logger.info("Starting Facet job")
        command_list = []
        command_list.append(f"(LatheEngraver Facet job)")
        command_list.append(f"(Min and Max values: {self.vMin}, {self.vMax} )")
        command_list.append(f"(Surface-to-B length: {self.tool_length})")
        command_list.append(f"(Tool diameter: {self.cutter_diam})")
        command_list.append(f"(Tool step-over: {self.step_over})")
        command_list.append(f"(Segments: {self.segments}, A rotation: {self.arotate})")
        command_list.append(f"(B angle range: {self.min_B} to {self.max_B})")
        command_list.append(f"(Fixed axis increment: {self.new_increment})")
        command_list.append(f"(B-angle smoothing points: {self.smooth_points})")
        command_list.append(f"(Ovality compensation: {self.do_oval})")
        data = dict(title="Writing Gcode...", text="Facet job is writing.", delay=600000, type="info")
        self.send_le_message(data)
        
        profile_points = self.resample_profile()
        self._logger.info(profile_points)
        # Preamble
        command_list.append("G21")
        command_list.append("G90")
        command_list.append("G94")

        reference_radius = self.diam / 2

        # PRECOMPUTE: spline values and radii for all profile points (vectorized)
        pp_arr = np.asarray(profile_points, dtype=float)
        z_vals = self.spline(pp_arr)  # vectorized CubicSpline evaluation
        radii_arr = reference_radius + (z_vals - self.referenceZ)
        max_radius = float(np.max(radii_arr))
        command_list.append(f"(Max radius: {max_radius})")

        # Max Z and scaling
        max_z = max_radius * (1 - math.cos(math.pi / self.segments))
        command_list.append(f"(Max Z: {max_z:0.2f})")
        max_z = max_z * self.depth_mod
        command_list.append(f"(Max Z with scaling: {max_z:0.2f})")
        if self.depth and max_z > self.depth:
            max_z = self.depth
            command_list.append(f"(Max Z limited to {self.depth:0.2f})")
        pass_info = divmod(max_z, self.step_down)
        passes = pass_info[0]
        last_pass_depth = pass_info[1]
        self._logger.info(f"Passes = {passes}, Last Pass = {last_pass_depth}")
        total_passes = int(passes + 1) if last_pass_depth > 0.0 else int(passes)
        '''
        # Angles
        tool_radius = self.cutter_diam / 2
        facet_angle = 360 / self.segments
        delta_theta = (tool_radius * self.step_over) / max_radius
        delta_degrees = math.degrees(delta_theta)
        start_a = delta_degrees
        end_a = facet_angle - delta_degrees
        #num_a_steps = int(math.ceil(facet_angle) / delta_degrees)
        num_a_steps = int((end_a - start_a) / delta_degrees) + 1
        self._logger.info(f"Depth passes: {total_passes}, Facet angle: {facet_angle}, Delta degrees: {math.degrees(delta_theta)}, Num A steps: {num_a_steps}")
        '''
        tool_radius = self.cutter_diam / 2
        facet_angle = 360 / self.segments
        offset_rad = tool_radius / max_radius
        offset_deg = math.degrees(offset_rad)

        # Start and end angles for the tool center
        start_a = offset_deg
        end_a = facet_angle - offset_deg
        #arc_span = end_a - start_a
        arc_span = facet_angle
        # Calculate num_a_steps as close as possible to what the user wants
        # (for example, based on initial step_over)
        initial_step_angle = math.degrees((tool_radius * self.step_over) / max_radius)
        num_a_steps = int(round(arc_span / initial_step_angle))
        if num_a_steps < 1:
            num_a_steps = 1

        # Now, recalculate step_over so that num_a_steps is integer and covers the arc evenly
        step_angle = arc_span / num_a_steps
        step_over = max_radius * math.radians(step_angle) / tool_radius
        self._logger.info(f"Step-over adjusted. Original: {self.step_over}, new: {step_over}")
        #added_steps = math.ceil(1/self.step_over)
        self.step_over = step_over
        delta_theta = (tool_radius * self.step_over) / max_radius
        delta_degrees = math.degrees(delta_theta)
        start_a = offset_deg
        num_a_steps = num_a_steps+1 #this adjusts for being off by one delta_theta
        self._logger.info(f"Depth passes: {total_passes}, Overall facet angle: {facet_angle}, Offset degrees: {offset_deg}, Delta degrees (step-over): {math.degrees(delta_theta)}, Num A steps: {num_a_steps}")


        # PRECOMPUTE: coords and B-derived trig at zero depth (reuse across all passes)
        coords_cache = [self.calc_coords(x) for x in profile_points]
        baseX = np.array([c["X"] for c in coords_cache], dtype=float)
        baseZ = np.array([c["Z"] for c in coords_cache], dtype=float)
        B_deg = np.array([c["B"] for c in coords_cache], dtype=float)
        B_rad_neg = np.deg2rad(-B_deg)
        sinB = np.sin(B_rad_neg)
        cosB = np.cos(B_rad_neg)

        # A-axis rotation per move
        seg_rot = self.arotate / (len(profile_points) - 1)
        self._logger.info(f"Segment rotation: {seg_rot}")
        A_rot = 360 / self.segments

        # Safe positions
        sign, safe = self.safe_retract()
        start = coords_cache[0]
        end = coords_cache[-1]
        # retract = +5 along normal at each point; compute when needed by index

        current_a = 0
        a_direction = 1
        a_measure = start_a
        ease_down = True
        completion = 20

        for j in range(0, self.segments):
            seg_start = time.time()
            data = dict(title="Facet Writing", text=f"Facet {j+1} of {self.segments} is writing", delay=completion*1000, type="info")
            self.send_le_message(data)
            facet_list = []
            a_direction = 1
            #facet_start_a = (facet_angle * j) + offset_deg
            facet_start_a = (facet_angle * j)
            command_list.append(f"(Starting facet {j+1} of {self.segments})")
            command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
            command_list.append(f"G0 B{start['B']:0.4f}")
            command_list.append(f"G0 X{(baseX[0] + 5 * sinB[0]):0.4f}")
            command_list.append(f"G0 Z{(baseZ[0] + 5 * cosB[0]):0.4f}")

            current_a = facet_start_a
            a_measure = current_a

            for a_step in range(num_a_steps):
                plunge = True
                section_done = False
                max_zmod = 0.0
                previous_depth = 0.0

                facet_list.append(f"(Facet A angle step {a_step+1} of {num_a_steps})")
                self._logger.debug(f"Facet angle step {a_step+1} of {num_a_steps}")
                facet_list.append(f"(Current A angle: {current_a:.3f})")
                facet_list.append(f"(A measure: {a_measure:.3f})")
                facet_list.append(f"G0 A{current_a:.3f}")

                for depth in range(1, total_passes + 1):
                    previous_coord = None
                    nominal_depth = depth * self.step_down  # positive value

                    if depth > 1:
                        self._logger.debug(f"Depth={depth}, max_zmod={max_zmod}, nominal={nominal_depth}, previous={previous_depth}")
                        if section_done:
                            continue
                        if nominal_depth >= max_zmod and previous_depth >= max_zmod:
                            self._logger.debug(f"Max Z mod reached at: {max_zmod:.2f}, stopping section")
                            section_done = True
                        if nominal_depth >= max_zmod or math.isclose(nominal_depth, max_zmod, abs_tol=0.02):
                            nominal_depth = max_zmod

                    if depth == 1:
                        thiscut = nominal_depth
                        self._logger.debug("First pass...")
                        section_done = False
                    else:
                        thiscut = nominal_depth - previous_depth

                    previous_depth = nominal_depth

                    if not section_done:
                        self._logger.debug(f"Cut depth on this pass: {thiscut}")
                        max_zmod = 0.0

                        # Decide index order once per pass to avoid reversed(list) cost
                        if a_direction == 1:
                            idx_iter = range(len(profile_points))
                        else:
                            idx_iter = range(len(profile_points) - 1, -1, -1)

                        i = 0
                        for idx in idx_iter:
                            # radius for this x
                            current_radius = max_radius if self.invert_facet else float(radii_arr[idx])

                            # angle relative within the facet
                            relative_a = a_step * math.degrees(delta_theta) + math.degrees(delta_theta)  # same as before
                            z_mod = self.sagitta_distance(math.radians(relative_a), current_radius)
                            
                            if self.invert_facet:
                                #self._logger.info(f"Step {a_step}, max_z: {max_z},z_mod: {z_mod},z_mod*d: {z_mod * self.depth_mod}")
                                z_mod = max_z - (z_mod * self.depth_mod)
                                
                            else:    
                                z_mod = z_mod * self.depth_mod

                            if z_mod > max_zmod:
                                max_zmod = z_mod
                                self._logger.debug(f"New max Z mod: {max_zmod:.2f}")

                            if z_mod > nominal_depth:
                                z_mod = nominal_depth
                                if depth == 1 and ease_down:
                                    fract = nominal_depth / 60.0
                                    z_mod = fract * (i + 1)
                                    facet_list.append(f"(Ease down step with z_mod: {z_mod:.2f})")
                                    if z_mod > nominal_depth:
                                        facet_list.append(f"(Ease down done)")
                                        z_mod = nominal_depth
                                        ease_down = False

                            # Adaptive feed scaling
                            if self.adaptive and thiscut < self.step_down:
                                scale = self.feedscale + (1.0 - self.feedscale) * (thiscut / self.step_down)
                                feed = (self.feed if previous_coord is None else self.calc_feedrate(previous_coord, {"X": baseX[idx], "Z": baseZ[idx], "B": B_deg[idx]})) * scale
                            else:
                                if previous_coord:
                                    feed = self.calc_feedrate(previous_coord, {"X": baseX[idx], "Z": baseZ[idx], "B": B_deg[idx]})
                                else:
                                    feed = self.feed

                            if self.do_oval:
                                # facet uses negative depth direction; subtract ovality
                                oval_mod = -self.ovality_mod(profile_points[idx], current_a)
                                z_mod = z_mod + oval_mod
                                
                            # tip position at depth (-z_mod) using cached sin/cos
                            trans_x = baseX[idx] + (-z_mod) * sinB[idx]
                            trans_z = baseZ[idx] + (-z_mod) * cosB[idx]

                            #facet_list.append(f"(minus z_mod: {z_mod:0.1f})")

                            a_move = current_a + (seg_rot * i * a_direction) if seg_rot else current_a
                            # plunge reduction only for first emitted move of this pass
                            if plunge:
                                feed = self.feed / 2.0
                            facet_list.append(f"G93 G1 X{trans_x:.3f} Z{trans_z:.3f} A{a_move:.3f} B{B_deg[idx]:.3f} F{feed:.1f}")
                            previous_coord = {"X": baseX[idx], "Z": baseZ[idx], "B": B_deg[idx]}
                            plunge = False
                            i += 1

                        a_direction *= -1
                    else:
                        # retract from last index used (use idx from previous loop if available)
                        # fall back to index 0 if not set
                        ridx = idx if 'idx' in locals() else 0
                        retract_x = baseX[ridx] + 5.0 * sinB[ridx]
                        retract_z = baseZ[ridx] + 5.0 * cosB[ridx]
                        facet_list.append(f"(Section done, retracting to safe position)")
                        facet_list.append(f"G0 X{retract_x:.3f} Z{retract_z:.3f} B{B_deg[ridx]:.3f}")

                    ease_down = False
                    if seg_rot:
                        current_a = a_move

                current_a += math.degrees(delta_theta)
                a_measure += math.degrees(delta_theta)
                # retract at end of a_step using last index
                ridx = idx if 'idx' in locals() else 0
                facet_list.append(f"G0 X{(baseX[ridx] + 5.0 * sinB[ridx]):.3f} Z{(baseZ[ridx] + 5.0 * cosB[ridx]):.3f} B{B_deg[ridx]:.3f}")

            completion = time.time() - seg_start
            self._logger.info(f"Facet completion time: {completion}")
            command_list.extend(facet_list)

            # rotate to next segment
            next_start = facet_angle * (j + 1)
            current_a = next_start
            a_measure = next_start
            command_list.append(f"G0 A{next_start:0.3f}")
        command_list.append("G94")
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
        self.send_le_clear()
        self._plugin_manager.send_plugin_message('latheengraver',  dict(type='filerefresh'))

    def generate_flute_job(self):
        self._logger.info("Starting Flute job")
        data = dict(title="Writing Gcode...", text="Flute job is writing.", delay=60000, type="info")
        self.send_le_message(data)

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
        command_list.append(f"(Fixed axis increment: {self.new_increment})")
        command_list.append(f"(B-angle smoothing points: {self.smooth_points})")
        
        profile_points = self.resample_profile()

        
        # Calculate depth passes
        pass_info = divmod(self.depth, self.step_down)
        passes = pass_info[0]
        last_pass_depth = pass_info[1]
        if last_pass_depth:
            total_passes = int(passes + 1)
        else:
            total_passes = int(passes)

        # Calculate A rotation per flute and per move (for helical flutes)
        flute_angle = 360 / self.segments
        seg_rot = self.arotate / (len(profile_points) - 1) if len(profile_points) > 1 else 0

        # Lead-in/lead-out calculations
        lead_in_x = lead_out_x = total_in_step = total_out_step = in_inc = out_inc = None
        if self.leadin or self.leadout:
            #try:
            if self.axis == "Z":
                #swap for Z
                lead_out_x = self.x_to_arc(profile_points, self.leadout, start=True)
                lead_in_x = self.x_to_arc(profile_points, -self.leadin, start=False)
            else:
                lead_in_x = self.x_to_arc(profile_points, self.leadin, start=True)
                lead_out_x = self.x_to_arc(profile_points, -self.leadout, start=False)
            self._logger.debug(f"lead_in_val: {lead_in_x}, lead_out_val: {lead_out_x}")

            if self.axis == "X":
                if lead_out_x < lead_in_x:
                    self._plugin_manager.send_plugin_message("latheengraver", dict(type="simple_notify",
                                                        title="Lead-in/Lead-out error",
                                                        text="Lead-in and lead-out overlap, please adjust values",
                                                        hide=True,
                                                        delay=10000,
                                                        notify_type="error"))
                    return

                total_in_step = int((lead_in_x - profile_points[0])/self.new_increment)
                total_out_step = int((profile_points[-1] - lead_out_x)/self.new_increment)
                in_inc = self.step_down/(total_in_step) if total_in_step else 0
                out_inc = self.step_down/(total_out_step) if total_out_step else 0
                self._logger.info(f"steps lead-in: {total_in_step}, lead-out {total_out_step}")
                self._logger.info(f"increment for lead-in: {in_inc}, lead-out {out_inc}")

            if self.axis == "Z":
                if lead_out_x > lead_in_x:
                    self._plugin_manager.send_plugin_message("latheengraver", dict(type="simple_notify",
                                                        title="Lead-in/Lead-out error",
                                                        text="Lead-in and lead-out overlap, please adjust values",
                                                        hide=True,
                                                        delay=10000,
                                                        notify_type="error"))
                    return
                self._logger.debug(f"Z, profile_points first/last: {profile_points[0]}, {profile_points[-1]}")
                total_in_step = abs(int((lead_in_x - profile_points[-1])/self.new_increment))
                total_out_step = abs(int((profile_points[0] - lead_out_x)/self.new_increment))
                in_inc = self.step_down/(total_in_step) if total_in_step else 0
                out_inc = self.step_down/(total_out_step) if total_out_step else 0
                self._logger.info(f"steps lead-in: {total_in_step}, lead-out {total_out_step}")
                self._logger.info(f"increment for lead-in: {in_inc}, lead-out {out_inc}")
            
        # Reverse the profile for Z axis
        #if self.axis == "Z":
        #    profile_points.reverse()
        
        if self.axis == "X":
            z_at_min = self.spline(self.vMin)
            z_at_max = self.spline(self.vMax)
            if z_at_max < z_at_min:
                profile_points.reverse()

        # Preamble
        command_list.append("G21")
        command_list.append("G90")
        command_list.append("G94")
        sign, safe = self.safe_retract()
        completion = 20
        for flute in range(self.segments):
            seg_start = time.time()
            data = dict(title="Flute Writing", text=f"Flute {flute+1} of {self.segments} is writing", delay=completion*1000, type="info")
            self.send_le_message(data)
            lastcut = False
            flute_dir = 1
            previous_depth = 0
            base_a = flute * flute_angle
            command_list.append(f"(Starting flute {flute+1} of {self.segments})")
            command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
            # Move to start position for this flute
            start = self.calc_coords(profile_points[0])
            trans_x, trans_z = self.cut_depth_value(start, 5)
            b_move  = (f"G0 B{start['B']:0.4f} A{base_a:.4f}")
            move_1 = (f"G0 X{trans_x:0.4f}")
            move_2 = (f"G0 Z{trans_z:0.4f}")
            if self.axis == "X":
                command_list.append(b_move)
                command_list.append(move_1)
                command_list.append(move_2)
            else:
                command_list.append(b_move)
                command_list.append(move_2)
                command_list.append(move_1)

            for current_pass in range(1, total_passes + 1):
                #NOTE, using negative depths here
                nominal_depth = current_pass * self.step_down * -1
                if current_pass == total_passes and last_pass_depth:
                    nominal_depth = self.depth * -1

                if current_pass == 1:
                    thiscut = nominal_depth
                else:
                    thiscut = nominal_depth - previous_depth
                previous_depth = nominal_depth

                if nominal_depth == self.depth * -1:
                    lastcut = True
                    command_list.append("(Last cut)")

                command_list.append(f"(Cut depth: {nominal_depth})")
                previous_coord = None

                # Alternate direction for each pass
                if flute_dir == 1:
                    points_iter = list(enumerate(profile_points))
                else:
                    points_iter = [(i, v) for i, v in zip(range(len(profile_points)-1, -1, -1), reversed(profile_points))]

                #TODO this is not working for lead-ins/outs!!!!
                for idx, each in points_iter:
                    # Calculate position from start and end for current direction
                    if flute_dir == 1:
                        position_from_start = idx
                        position_from_end = len(profile_points) - 1 - idx
                        leadin_check = position_from_start
                        leadout_check = position_from_end
                    else:
                        position_from_start = len(profile_points) - 1 - idx
                        position_from_end = idx
                        # SWAP lead-in and lead-out checks for reverse pass
                        leadin_check = position_from_end
                        leadout_check = position_from_start

                    depth = nominal_depth
                    if self.leadin and total_in_step and leadin_check < total_in_step:
                        depth = self.lead_calc("in", nominal_depth, total_in_step - leadin_check, in_inc)
                        command_list.append(f"(lead in, index:{leadin_check} inc:{in_inc} depth:{depth})")
                    if self.leadout and total_out_step and leadout_check < total_out_step:
                        depth = self.lead_calc("out", nominal_depth, total_out_step - leadout_check, out_inc)
                        command_list.append(f"(lead out, index:{leadout_check} inc:{out_inc} depth:{depth})")

                    coord = self.calc_coords(each)
                    if previous_coord:
                        feed = self.calc_feedrate(previous_coord, coord)
                    else:
                        feed = self.feed

                    previous_coord = coord

                    if seg_rot:
                        current_a = base_a + seg_rot * idx
                    else:
                        current_a = base_a

                    if self.do_oval:
                        self._logger.debug(f"Pre-oval depth at {current_a}: {depth}")
                        depth_mod = self.ovality_mod(each, current_a)
                        depth = depth + depth_mod
                        self._logger.debug(f"Post-oval depth at {current_a}: {depth}")

                    if self.adaptive:
                        # Scale feed between maxscale (shallow cut) and 1.0 (full step_down)
                        if abs(thiscut) < self.step_down:
                            scale = (
                                self.feedscale+
                                (1.0 - self.feedscale) * (abs(thiscut) / self.step_down)
                            )
                            self._logger.debug(f"feed adjust from {feed} to {feed*scale}")
                            feed = feed * scale

                    trans_x, trans_z = self.cut_depth_value(coord, depth)
                    command_list.append(
                        f"G93 G1 X{trans_x:.3f} Z{trans_z:.3f} A{current_a:.3f} B{coord['B']:.3f} F{feed:.1f}"
                    )
                # Retract at end of pass
                trans_x, trans_z = self.cut_depth_value(coord, 5)
                command_list.append(f"G0 X{trans_x:.3f} Z{trans_z:.3f} B{coord['B']:.3f}")
                if not self.conventional:
                    flute_dir *= -1  # Reverse direction for next pass
                else:
                    if not lastcut:
                        #move back to start
                        command_list.append(f"G0 {safe}{sign}{self.clearance+10:0.3f}")
                        if self.axis == "X":
                            command_list.append(b_move)
                            command_list.append(move_1)
                            command_list.append(move_2)
                        else:
                            command_list.append(b_move)
                            command_list.append(move_2)
                            command_list.append(move_1)
            completion = time.time() - seg_start
            self._logger.info(f"Flute written in {completion}")
            # After all passes for this flute, rotate to next flute
            command_list.append(f"G0 A{(base_a + flute_angle):.3f}")
            #command_list.append("G92 A0")
        command_list.append("G94")
        command_list.append("M5")
        command_list.append("M30")
        output_name = self.name.removesuffix(".txt")
        output_name = f"Flute_S{self.segments}_Arot{self.arotate}_D{self.depth}_"+output_name+".gcode"
        path_on_disk = "{}/{}".format(self._settings.getBaseFolder("watched"), output_name)

        with open(path_on_disk,"w") as newfile:
            for line in command_list:
                newfile.write(f"\n{line}")

        self.send_le_clear()
        self._plugin_manager.send_plugin_message('latheengraver',  dict(type='filerefresh'))

    def generate_wrap_job(self):
        #TODO: It makes  more sense to go back and write a parser explicity for this.
        # GcodeRipper works, but is quite clunky

        #create profile from diameter reference
        profile_points = []
        data = dict(title="Writing Gcode...", text="Wrap job is writing.", delay=60000, type="info")
        self.send_le_message(data)
        #truncate profile beween vMin and vMax
        profile_points = self.resample_profile()

        #gcr = G_Code_Rip.G_Code_Rip()
        basefolder = self._settings.getBaseFolder("uploads")
        self.gcr.Read_G_Code(f"{basefolder}/{self.selected_file}", XYarc2line=True, units="mm")
        #profile name
        #self._logger.debug(self.gcr.g_code_data)
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

        #realizing that this is correct for the first
        
        a = target_x-self.vMin

        xtoscale = (a)/self.width
        self._logger.info(f"Target X-value={target_x}, xtoscale={xtoscale}")
        #have to go back and handle z cases too
        temp,minx,maxx,miny,maxy,minz,maxz  = self.gcr.scale_rotate_code(self.gcr.g_code_data,
                                                                    [xtoscale,sf,1,1],
                                                                    0,
                                                                    split_moves=True,
                                                                    min_seg_length=self.new_increment)
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
                                        self.smooth_points,
                                        self.do_oval,
                                        plugin=self)
                                        
        #self._logger.info(temp)
        #get first X and Z moves that are not complex
        first_x = None
        first_z = None
        for line in temp:
            if line[0] == 0 or line[0] == 1:
                if not isinstance(line[1][3], complex):
                    first_z = line[1][2]
                if not isinstance(line[1][1], complex):
                    first_x = line[1][1]
                    break


        path_on_disk = "{}/{}".format(self._settings.getBaseFolder("watched"), output_name)
        arots = 0
        if self.segments > 1:
            arots = 360/self.segments
        repeats = self.segments
        sign, safe = self.safe_retract()
        gcode = []
        gcode.append(f"(LatheEngraver G-code Wrapping)")
        gcode.append(f"(Ref. Diam: {self.diam}, Projection: {self.radius_adjust}, G-code width: {self.width})")
        gcode.append(f"(Profile distance: {profile_dist:0.2f}, X-scale factor: {xtoscale}, A-scale factor: {sf})")
        gcode.append("(Safe moves added)")
        gcode.extend([f"G90 G21",f"G0 {safe}{sign}{10+self.clearance:0.4f}",f"G0 X{first_x:0.2f}"])
        a_offset = 0
        first_move = False
        for j in range(0, self.segments):
            a_offset=arots*(j)
            for line in self.gcr.generategcode(temp,
                                                Rstock=self.diam/2,
                                                no_variables=True,
                                                Wrap="SPECIAL",
                                                FSCALE="None",
                                                do_oval=self.do_oval,
                                                a_offset=a_offset):
                if j < self.segments and line.startswith("M30"):
                    continue
                else:
                    if not first_move and line.startswith("G0"):
                        #parse to see if this is a G0 that goes to some X and Z value
                        x,z,b = self._parse_g0(line)
                        if x is None or z is None:
                            pass
                        else:
                            first_move = True
                            if b is not None:
                                gcode.append(f"G0 B{b}")
                            gcode.extend([f"G0 X{x}",f"G0 Z{z}"])
                    gcode.append(f"{line}")
            gcode.append("G0 A0")
            next_a = a_offset+arots
            gcode.append(f"G0 A{next_a:0.4f}")

        with open(path_on_disk,"w") as newfile:
            for line in gcode:
                newfile.write(f"\n{line}")

        self.send_le_clear()
        self._plugin_manager.send_plugin_message('latheengraver',  dict(type='filerefresh'))

    def send_le_message(self, data):
        
        payload = dict(
            type="simple_notify",
            title=data["title"],
            text=data["text"],
            hide=True,
            delay=data["delay"],
            notify_type=data["type"]
        )

        self._plugin_manager.send_plugin_message("latheengraver", payload)

    def send_le_clear(self):
        self._plugin_manager.send_plugin_message("latheengraver", dict(type='clear_all'))
        self._logger.debug("Cleared notification")

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
            self.new_increment = float(data["steps"])
            if self.new_increment != self.increment:
                self._logger.info("Increment has changed, resampling")
            self.smooth_points = int(data["smoothing"])
            if self.segments == 0:
                self.segments = 1
            self.vMax = float(data["vMax"])
            self.vMin = float(data["vMin"])
            self.feed = int(data["feed"])
            self.risky_clearance = bool(data["risky"])
            self.conventional = bool(data["conventional"])
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

            if self.do_oval:
                if bool(data["ignore_oval"]):
                    self.do_oval = False
                    self._logger.debug("Ignoring ovality measurements")
                else:
                    self.create_a_spline()

            if self.mode == "laser":
                self.test = bool(data["test"])
                self.power = int(data["power"])

                #self.start_max = bool(data["start"])
                self.generate_laser_job()
                return

            if self.mode == "flute":
                self.depth = float(data["depth"])
                self.step_down = float(data["step_down"])
                self.leadin = float(data["leadin"])
                self.leadout = float(data["leadout"])
                self.adaptive = bool(data["adaptive"])
                self.feedscale = float(data["feedscale"])
                self.generate_flute_job()
                return
            
            if self.mode == "wrap":
                self.referenceZ = float(data["refZ"])
                self.width = float(data["width"]) #width in X as reported by bgs
                self.selected_file = data["filename"]["path"]
                self.diam = float(data["diam"])
                self.radius_adjust = bool(data["radius_adjust"])
                self.singleB = bool(data["singleB"])
                self.generate_wrap_job()
                return
            
            if self.mode == "facet":
                self.referenceZ = float(data["refZ"])
                self.diam = float(data["diam"])
                self.step_down = float(data["step_down"])
                self.cutter_diam = float(data["tool_diam"])
                self.step_over = float(data["step_over"])
                self.invert_facet = bool(data["facet_invert"])
                self.depth_mod = float(data["depth_mod"])
                self.depth = float(data["depth"])
                self.adaptive = bool(data["adaptive"])
                self.feedscale = float(data["feedscale"])

                if self.segments < 3:
                    self._plugin_manager.send_plugin_message("latheengraver", dict(type="simple_notify",
                                                                    title="Facet Error",
                                                                    text="Minimum segments for Facet is 3",
                                                                    hide=True,
                                                                    delay=10000,
                                                                    notify_type="error"))
                    return
                if self.axis != 'X':
                    self._plugin_manager.send_plugin_message("latheengraver", dict(type="simple_notify",
                                                                    title="Facet Error",
                                                                    text="Facets are currently only compatible with X-axis scans.",
                                                                    hide=True,
                                                                    delay=10000,
                                                                    notify_type="error"))
                    return
                self.generate_facet_job()
                return
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
            self.smooth_points = int(data["smoothing"])
            getB = bool(data["getB"])
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
            gcode = ["G90","G21","G94",f"G0 {safe}{sign}{10+self.clearance:0.4f}"]
            coord = self.calc_coords(self.target)
            if getB:
                self._logger.info(f"Calculated B: {coord['B']}")
                msg = dict(title="Coordinates at target", text="Calculated B: {0:0.2f}<br>Calculated X: {1:0.2f}<br>Calculated Z: {2:0.2f}".format(coord['B'], coord['X'], coord['Z']), type="info", delay=10000)
                self.send_le_message(msg)
                return
            else:
                b_move  = (f"G0 B{coord['B']:0.4f}")
                move_1 = (f"G0 X{coord['X']:0.4f}")
                move_2 = (f"G0 Z{coord['Z']:0.4f}")
                if self.axis == "X":
                    gcode.append(b_move)
                    gcode.append(move_1)
                    gcode.append(move_2)
                else:
                    gcode.append(b_move)
                    gcode.append(move_2)
                    gcode.append(move_1)
                gcode.append("G94")
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

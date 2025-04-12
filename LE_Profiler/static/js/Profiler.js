$(function() {
    function ProfilerViewModel(parameters) {
        var self = this;
        self.global_settings = parameters[1];
        self.xValues = [];
        self.zValues = [];
        self.vMax = null;
        self.vMin = null;
        self.target_position = []; 
        self.smoothedZValues = [];  // Store smoothed Z values
        self.annotations = [];
        self.markerAction = ko.observable("zeroPoint");
        self.tool_length = ko.observable(135);
        self.min_B = ko.observable(-180);
        self.max_B = ko.observable(180);
        self.start_max = ko.observable(0);
        self.steps = ko.observable(1.0);
        self.smoothing = ko.observable(6);
        self.side = ko.observable("front");
        self.Arot = ko.observable(0);
        self.depth = ko.observable(1);
        self.step = ko.observable(1);
        self.leadin = ko.observable(0);
        self.leadout = ko.observable(0);
        self.smooth_points = ko.observable(4);
        self.increment = ko.observable(0.5);
        self.reversed = false;
        self.isZFile = false;
        self.isXFile = false;
        self.name = null;
        self.pd = null;

        self.wrapfiles = null;
        self.scans = null;

        //Laser
        self.power = ko.observable(250);
        self.feed = ko.observable(200);
        self.test = ko.observable(0);
        self.segments = ko.observable(1);
        //Fluting/wrapping
        self.scale = false;
        self.refdiam = ko.observable(0);
        self.refset = null;
        self.referenceZ = null;
        self.width = ko.observable(0);
        self.selectedGCodeFile = null;
        self.radius_adjust = ko.observable(0);
        self.singleB = ko.observable(0);

        self.mode = ko.observable("none");
        
        self.onModeChange = function () {
            if (self.mode() === "wrap") {
                $(".laser").hide();
                $(".wrap").show();
                $(".flute").hide();
                self.fetchWrapFiles(); // Fetch GCode files for wrap mode
            } else if (self.mode() === "laser") {
                $(".laser").show();
                $(".wrap").hide();
                $(".flute").hide();
            } else if (self.mode() === "flute") {
                $(".laser").hide();
                $(".wrap").hide();
                $(".flute").show();
            }
        }

        // Fetch the list of .txt files from the uploads/scans directory
        self.fetchProfileFiles = function() {
            OctoPrint.files.listForLocation("local/scans", false)
                .done(function(data) {
                    var scans = data.children;
                    console.log(scans);
                    scans.sort((a,b) => { return a.name.localeCompare(b.name) });
                    self.scans = scans;
                    populateFileSelector(scans, "#scan_file_select", "machinecode");
                })
                .fail(function() {
                    console.error("Failed to fetch GCode files.");
                });
        };

        self.fetchWrapFiles = function() {
            OctoPrint.files.listForLocation("local/wrap", false)
                .done(function(data) {
                    var files = data.children;
                    console.log(files);
                    files.sort((a,b) => { return a.name.localeCompare(b.name) });
                    self.wrapfiles = files;
                    populateFileSelector(files, "#wrapFileSelect", "gcode");
                })
                .fail(function() {
                    console.error("Failed to fetch GCode files.");
                });
        };

        function populateFileSelector(files, elem, type) {
            var fileSelector = $(elem);
            fileSelector.empty();
            fileSelector.append($("<option>").text("Select file").attr("value", ""));
            files.forEach(function(file, i) {
                var option = $("<option>")
                    .text(file.display)
                    .attr("value", file.name)
                    .attr("download",file.refs.download)
                    .attr("path",file.path)
                    .attr("index", i);
                fileSelector.append(option);
            });
        }

        self.onBeforeBinding = function () {
            self.settings = self.global_settings.settings.plugins.Profiler;
            //console.log(self.global_settings);
            self.fetchProfileFiles();
            $(".laser").hide();
            $(".wrap").hide();
            $(".zscan").hide();

            self.smooth_points = self.settings.smooth_points;
            self.tool_length = self.settings.tool_length;
            self.increment = self.settings.increment;

        };

        // Bind mode change event
        $("#modeSelect").on("change", function () {
            self.mode($(this).val());
            self.onModeChange();
            console.log(self.mode());
        });

        self.do_distance = function() {
            if (!self.isZFile && self.mode() === "wrap" && self.vMax && self.vMin)  {
                self.pd = self.get_pd();
                return true;
            }
        }
        // Function to plot the profile using Plotly
        function plotProfile(isZFile) {

            var trace = {
                x: self.xValues,
                y: self.zValues,
                mode: 'lines',
                name: 'Profile',
                line: {
                    color: 'blue',
                    width: 2
                }
            };

            var layout = {
                title: 'Profile Plot',
                xaxis: { 
                    title: 'X Axis',
                    scaleanchor: 'y',  // Ensure equal scaling
                    scaleratio: 1
                },
                yaxis: { 
                    title: 'Z Axis',
                    //scaleanchor: 'x',  // Equal scaling with X axis
                    scaleratio: 1,
                    autorange: 'reversed'
                },
                annotations: self.annotations,  // Include any annotations (tags)
                showlegend: false
            };

            // Plot the data
            Plotly.newPlot('profilePlot', [trace], layout)
            .then(function() {
                // Ensure plotly_click is bound AFTER the plot is rendered
                document.getElementById('profilePlot').on('plotly_click', function (data) {
                    if (data && data.points && data.points.length > 0) {
                        var clickedPoint = data.points[0];
                        var clickedX = clickedPoint.x;
                        var clickedZ = clickedPoint.y;
                
                        if (self.markerAction() === "zeroPoint") {
                            // Normalize both X and Z axes
                            self.xValues = self.xValues.map(x => x - clickedX);
                            self.zValues = self.zValues.map(z => z - clickedZ);
                            self.annotations = []; // Clear all annotations
                            self.vMin = null;
                            self.vMax = null;
                            self.target_position = null;
                            plotProfile(self.isZFile); // Replot based on file mode
                            //Plotly.react('profilePlot');
                        } else if (self.isZFile) {
                            // Z-file mode: Handle Z-axis selections
                            if (self.markerAction() === "Max") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Max'));
                                if (self.vMin && clickedZ < self.vMin) {
                                    alert("Max must be greater than Min");
                                    return;
                                }
                                self.vMax = clickedZ;
                                self.annotations.push({
                                    x: clickedX,
                                    y: clickedZ,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Max: '+self.vMax,
                                    showarrow: true,
                                    arrowhead: 2,
                                    ax: 30,
                                    ay: -30
                                });
                                plotProfile(true);
                            } else if (self.markerAction() === "Min") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Min'));
                                if (self.vMax && clickedZ > self.vMax) {
                                    alert("Min must be less than Max");
                                    return;
                                }
                                self.vMin = clickedZ;
                                self.annotations.push({
                                    x: clickedX,
                                    y: clickedZ,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Min: '+self.vMin,
                                    showarrow: true,
                                    arrowhead: 2,
                                    ax: -30,
                                    ay: -30
                                });
                                plotProfile(true);

                            }  else if (self.markerAction() === "targetPoint") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Target'));
                                self.target_position = clickedZ;
                                self.annotations.push({
                                    x: clickedX,
                                    y: clickedZ,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Target: '+self.target_position,
                                    showarrow: true,
                                    arrowhead: 2,
                                    ax: 20,
                                    ay: 20
                                });
                                plotProfile(true);
                            }
                        } else if (self.isXFile) {
                            // X-file mode: Handle X-axis selections
                            if (self.markerAction() === "Max") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Max'));
                                if (self.vMin && clickedX < self.vMin) {
                                    alert("Max must be greater than Min");
                                    return;
                                }
                                self.vMax = clickedX;
                                self.annotations.push({
                                    x: clickedX,
                                    y: clickedZ,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Max: '+self.vMax,
                                    showarrow: true,
                                    arrowhead: 2,
                                    ax: 30,
                                    ay: -30
                                });
                                if (!self.do_distance()) { plotProfile(false); }
                            } else if (self.markerAction() === "Min") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Min'));
                                if (self.vMax && clickedX > self.vMax) {
                                    alert("Min must be less than Max");
                                    return;
                                }
                                self.vMin = clickedX;
                                self.annotations.push({
                                    x: clickedX,
                                    y: clickedZ,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Min: '+self.vMin,
                                    showarrow: true,
                                    arrowhead: 2,
                                    ax: -30,
                                    ay: -30
                                });
                                if (!self.do_distance()) { plotProfile(false); }
                            } else if (self.markerAction() === "targetPoint") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Target'));
                                self.target_position = clickedX;
                                self.annotations.push({
                                    x: clickedX,
                                    y: clickedZ,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'Target: '+self.target_position,
                                    showarrow: true,
                                    arrowhead: 2,
                                    ax: 20,
                                    ay: 20
                                });
                                plotProfile(false);
                            }
                            else if (self.markerAction() === "refset") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('D'));
                                self.referenceZ = clickedZ;
                                self.annotations.push({
                                    x: clickedX,
                                    y: clickedZ,
                                    xref: 'x',
                                    yref: 'y',
                                    text: 'D='+self.refdiam(),
                                    showarrow: true,
                                    arrowhead: 2,
                                    ax: 0,
                                    ay: 40
                                });
                                plotProfile(false);
                            }
                        }
                    }
                });
            });
        }

        $("#wrapFileSelect").on("change", function () {
            var filePath = $("#wrapFileSelect option:selected").attr("download");
            if (!filePath) return;
            var theindex = $("#wrapFileSelect option:selected").attr("index");
            var bgs_width = self.wrapfiles[theindex]["bgs_width"];
            self.selectedGCodeFile = self.wrapfiles[theindex];
            self.width = bgs_width;
        });

        // When a file is selected, load and plot the profile
        $("#scan_file_select").on("change", function () {
            var filePath = $("#scan_file_select option:selected").attr("path");
            self.name = $("#scan_file_select option:selected").attr("value");
            console.log(filePath);
            if (!filePath) return;
        
            // Determine the mode based on the file name
            self.isZFile = $("#scan_file_select option:selected").text().startsWith("Z");
            self.isXFile = $("#scan_file_select option:selected").text().startsWith("X");
            if (!$("#scan_file_select option:selected").text().endsWith("txt")){
                console.log("Not a txt file");
                alert("Selected file is not a text scan file.");
                return
            }

            if (self.isZFile) {
                $(".zscan").show();
            }
            else {
                $(".zscan").hide();
            }

            self.annotations = [];
            self.vMax = null;
            self.vMin = null;
            self.target_position = null;

            // Send the file info off
            self.createGraph(filePath);
            
        });
        

        // Button click event to reverse the Z values and replot
        $("#reverseZButton").on("click", function () {
            var fileName = $("#scan_file_select option:selected").attr("value");

            //console.log(fileName.charAt(0));
            if (self.isZFile) {
                self.zValues.reverse();
                //self.zValues = self.zValues.map(z => z * -1); 
                plotProfile(true);
            }
            else {
                self.xValues.reverse(); // Reverse the X values
                plotProfile(false);

            }
            //plotProfile();     // Replot with the reversed Z values
        });

        self.getPointsInRange = function() {
            var pointsInRange = [];
            for (var i = 0; i < self.xValues.length; i++) {
                pointsInRange.push({ x: parseFloat(self.xValues[i]).toFixed(3), z: parseFloat(self.zValues[i]).toFixed(3) });
            }
            return pointsInRange;
        };

        self.onDataUpdaterPluginMessage = function(plugin, data) {
            if (plugin == 'Profiler' && data.type == 'graph' && data.axis == 'X') {
                self.xValues = data.probe.map(point => point[0]);
                self.zValues = data.probe.map(point => point[1]);
                plotProfile(self.isZFile);
            }

            if (plugin == 'Profiler' && data.type == 'graph' && data.axis == 'Z') {
                self.xValues = data.probe.map(point => point[1]);
                self.zValues = data.probe.map(point => point[0]);
                plotProfile(self.isZFile);
            }

            if (plugin == 'Profiler' && data.type == 'distance') {
                self.pd = data.pd;
                self.annotations = self.annotations.filter(a => !a.text.startsWith('Width'));
                self.annotations.push({
                    x: 0,
                    y: 1,
                    xref: 'paper',
                    yref: 'paper',
                    text: 'Width: '+self.width+'<br>Pro. Dist.: '+self.pd,
                    showarrow: false,
                });
                console.log(self.pd);
                plotProfile(self.isZFile);
            }
        }

        //transmit the file path. It will be procesed and data sent back
        self.createGraph = function(filePath) {
            var data = {
                filepath: filePath
            };

            OctoPrint.simpleApiCommand("Profiler", "creategraph", data)
                .done(function(response) {
                    console.log("Graph info transmitted");
                })
                .fail(function() {
                    console.error("Graph info not transmitted");
                });

        };


        self.get_pd = function() {
            var data = {
                vMin: self.vMin,
                vMax: self.vMax
            };

            OctoPrint.simpleApiCommand("Profiler", "get_arc_length", data)
                .done(function(response) {
                    console.log("Info for arc length sent");
                })
                .fail(function() {
                    console.error("Did not get arc length");
                });

        };

        self.writeGCode = function() {
            //Data sanity checking
            if (self.mode() == "none") {
                alert("Mode must be set to write a job.");
                return;
            }

            if (!self.vMax || !self.vMin) {
                alert("Min. and Max. values must be set.");
                return;
            }

            if (self.isZFile) {
                var clearance = Math.max(...self.xValues);
            }
            else {
                var clearance = Math.max(...self.zValues);
            }

            var plot = self.getPointsInRange();
            //console.log(plot);
            var data = {
                plot_data: plot,
                mode: self.mode(),
                tool_length: self.tool_length(),
                max_B: self.max_B(),
                min_B: self.min_B(),
                power: self.power(),
                feed: self.feed(),
                test: self.test(),
                segments: self.segments(),
                vMax: self.vMax,
                vMin: self.vMin,
                filename: self.selectedGCodeFile,
                diam: self.refdiam(),
                clear: clearance,
                refZ: self.referenceZ,
                arotate: self.Arot(),
                side: self.side(),
                name: self.name,
                depth: self.depth(),
                step: self.step(),
                leadin: self.leadin(),
                leadout: self.leadout(),
                width: self.width,
                radius_adjust: self.radius_adjust(),
                singleB: self.singleB(),
                steps: self.steps(),
                smoothing: self.smoothing(),

            };
    
            OctoPrint.simpleApiCommand("", "write_job", data)
                .done(function(response) {
                    console.log("GCode written successfully.");
                })
                .fail(function() {
                    console.error("Failed to write GCode.");
                });
        };

        self.gotoposition = function() {
            //Data sanity checking
            if (self.isZFile && self.side == "none") {
                alert("Tool direction must be set for Z scans");
                return;
            }
    
            if (self.isZFile) {
                if (self.side === "back") {
                    var clearance = Math.abs(Math.min(...self.xValues));
                }
                else {
                    var clearance = Math.max(...self.xValues);
                }
            }

            else {
                var clearance = Math.max(...self.zValues);
            }

            var plot = self.getPointsInRange();
            var data = {
                plot_data: plot,
                target: self.target_position,
                tool_length: self.tool_length(),
                max_B: self.max_B(),
                min_B: self.min_B(),
                clear: clearance,
                side: self.side(),
                mode: "target",
            };
            console.log(data);
            OctoPrint.simpleApiCommand("Profiler", "go_to_position", data)
                .done(function(response) {
                    console.log("Go to target successful.");
                })
                .fail(function() {
                    console.error("Failed to go to target");
                });
        };
    }

    OCTOPRINT_VIEWMODELS.push({
        construct: ProfilerViewModel,
        dependencies: ["loginStateViewModel", "settingsViewModel"],
        elements: ["#tab_plugin_Profiler","#settings_plugin_Profiler"]
    });
});

$(function() {
    function ProfilerViewModel(parameters) {
        var self = this;
        self.global_settings = parameters[1];
        self.xValues = [];
        self.zValues = [];
        self.vMax = null;
        self.vMin = null;
        self.target_position = []; 
        self.smoothedZValues = [];
        self.annotations = [];
        self.markerAction = ko.observable("zeroPoint");
        self.tool_length = ko.observable(0.0);
        self.min_B = ko.observable(-180);
        self.max_B = ko.observable(180);
        self.start_max = ko.observable(0);
        self.smoothing = ko.observable(6);
        self.side = ko.observable("front");
        self.Arot = ko.observable(0);
        self.depth = ko.observable(0);
        self.step_down = ko.observable(1);
        self.leadin = ko.observable(0);
        self.leadout = ko.observable(0);
        self.smooth_points = ko.observable(4);
        self.increment = ko.observable(0.25);
        self.adaptive = ko.observable(0);
        self.feedscale = ko.observable(1.0);
        self.fpass = ko.observable(200);
        self.ignore_oval = ko.observable(0);
        self.conventional = ko.observable(0);
        self.reversed = false;
        self.isZFile = false;
        self.isXFile = false;
        self.name = null;
        self.pd = null;
        self.wrapfiles = null;
        self.scans = null;
        self.exp = ko.observable(false);

        // Laser
        self.laser_mode = ko.observable(false);
        self.power = ko.observable(250);
        self.feed = ko.observable(200);
        self.test = ko.observable(0);
        self.segments = ko.observable(1);
        self.laser_sections = ko.observableArray([]);
        self.polarAngles = ko.observableArray([]);
        self.polarSymmetry = ko.observable(1);

        // Color palette for committed layers
        var LAYER_COLORS = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
            '#ff7f00', '#a65628', '#f781bf', '#8dd3c7'
        ];

        // Polar plot shell constants
        var WORK_R     = 1.0;   // radius of the working (inner) area
        var SHELL_W    = 0.1;   // radial width of each committed-layer shell
        var MAX_SHELLS = 10;    // max committed layers; total radial extent = WORK_R + MAX_SHELLS * SHELL_W = 2.0

        // Fluting/wrapping
        self.scale = false;
        self.refdiam = ko.observable(0);
        self.refset = null;
        self.referenceZ = null;
        self.width = ko.observable(0);
        self.selectedGCodeFile = null;
        self.radius_adjust = ko.observable(0);
        self.singleB = ko.observable(0);
        self.risky = ko.observable(0);
        self.flute_gap = ko.observable(0);
        // Facet
        self.tool_diam = ko.observable(6.35);
        self.step_over = ko.observable(0.5);
        self.facet_invert = ko.observable(0);
        self.depth_mod = ko.observable(1.0);
        self.selectedSVGFile = null;
        self.svgfiles = null;

        self.extra_depth = ko.observable(0.0);
        self.mode = ko.observable("none");

        function toggleSection(targetMode) {
            $(".mode-section").hide();
            $(`.${targetMode}`).each(function () {
                const isExpOnly = $(this).hasClass("exp");
                if (!isExpOnly || !!self.exp) {
                    $(this).show();
                }
            });
        }

        self.onModeChange = function () {
            const mode = self.mode();
            toggleSection(mode);
            if (mode === "wrap") {
                self.fetchWrapFiles();
            } else if (mode === "facet") {
                self.fetchsvgFiles();
            } else if (mode === "flute") {
                self.fetchsvgFiles();
            }
        };

        self.fetchProfileFiles = function() {
            OctoPrint.files.listForLocation("local/scans", false)
                .done(function(data) {
                    var scans = data.children || [];
                    scans = scans
                        .filter(f => f.name && f.name.toLowerCase().endsWith(".txt"))
                        .sort((a, b) => a.name.localeCompare(b.name));
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
                    files.sort((a,b) => { return a.name.localeCompare(b.name) });
                    self.wrapfiles = files;
                    populateFileSelector(files, "#wrapFileSelect", "gcode");
                })
                .fail(function() {
                    console.error("Failed to fetch GCode files.");
                });
        };

        self.fetchsvgFiles = function() {
            OctoPrint.files.listForLocation("local/scans", false)
                .done(function(data) {
                    var files = data.children;
                    files = files
                        .filter(f => f.name && f.name.toLowerCase().endsWith(".svg"))
                        .sort((a,b) => { return a.name.localeCompare(b.name) });
                    self.svgfiles = files;
                    populateFileSelector(files, "#svgFileSelect", "machinecode");
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
                    .attr("download", file.refs.download)
                    .attr("path", file.path)
                    .attr("index", i);
                fileSelector.append(option);
            });
        }

        self.onBeforeBinding = function () {
            self.settings = self.global_settings.settings.plugins.profiler;
            self.fetchProfileFiles();
            $(".laser").hide();
            $(".wrap").hide();
            $(".zscan").hide();
            self.smoothing = self.settings.smooth_points;
            self.increment = self.settings.increment;
            self.exp = self.settings.exp();
        };

        $("#modeSelect").on("change", function () {
            self.mode($(this).val());
            self.onModeChange();
            console.log(self.mode());
        });

        self.do_distance = function() {
            if (!self.isZFile && self.mode() === "wrap" && self.vMax != null && self.vMin != null)  {
                self.pd = self.get_pd();
                return true;
            }
        };

        // ── Profile plot ──────────────────────────────────────────────────────
        function plotProfile(isZFile) {
            var trace = {
                x: self.xValues,
                y: self.zValues,
                mode: 'lines',
                //name: 'Profile',
                line: { color: 'blue', width: 2 }
            };

            // Build a colored rect shape for each committed layer
            var shapes = [];
            self.laser_sections().forEach(function(layer) {
                if (isZFile) {
                    // Z-file: range is along the Y axis
                    shapes.push({
                        type: 'rect',
                        xref: 'paper', yref: 'y',
                        x0: 0, x1: 1,
                        y0: Math.min(layer.vMin, layer.vMax),
                        y1: Math.max(layer.vMin, layer.vMax),
                        fillcolor: layer.color,
                        opacity: 0.25,
                        line: { width: 1, color: layer.color }
                    });
                } else {
                    // X-file: range is along the X axis
                    shapes.push({
                        type: 'rect',
                        xref: 'x', yref: 'paper',
                        x0: Math.min(layer.vMin, layer.vMax),
                        x1: Math.max(layer.vMin, layer.vMax),
                        y0: 0, y1: 1,
                        fillcolor: layer.color,
                        opacity: 0.25,
                        line: { width: 1, color: layer.color }
                    });
                }
            });

            var layout = {
                xaxis: {
                    //title: 'X Axis',
                    scaleanchor: 'y',
                    scaleratio: 1
                },
                yaxis: {
                    //title: 'Z Axis',
                    scaleratio: 1,
                    autorange: 'reversed'
                },
                annotations: self.annotations,
                shapes: shapes,
                showlegend: false,
                margin: { t: 20, b: 20, l: 20, r: 10 },
            };

            var config = {
                displayModeBar: false,
                showlegend: false,
            }

            Plotly.newPlot('profilePlot', [trace], layout, config)
            .then(function() {
                document.getElementById('profilePlot').on('plotly_click', function (data) {
                    if (data && data.points && data.points.length > 0) {
                        var clickedPoint = data.points[0];
                        var clickedX = clickedPoint.x;
                        var clickedZ = clickedPoint.y;
                        if (self.markerAction() === "zeroPoint") {
                            self.xValues = self.xValues.map(x => x - clickedX);
                            self.zValues = self.zValues.map(z => z - clickedZ);
                            self.annotations = [];
                            self.vMin = null;
                            self.vMax = null;
                            self.target_position = null;
                            plotProfile(self.isZFile);
                        } else if (self.isZFile) {
                            if (self.markerAction() == "Max") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Max'));
                                if (self.vMin != null && Number(clickedZ) < Number(self.vMin)) {
                                    alert("Max must be greater than Min");
                                    return;
                                }
                                self.vMax = Number(clickedZ);
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'Max: '+self.vMax, showarrow: true, arrowhead: 2, ax: 30, ay: -30 });
                                plotProfile(true);
                            } else if (self.markerAction() === "Min") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Min'));
                                if (self.vMax != null && Number(clickedZ) > Number(self.vMax)) {
                                    alert("Min must be less than Max");
                                    return;
                                }
                                self.vMin = Number(clickedZ);
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'Min: '+self.vMin, showarrow: true, arrowhead: 2, ax: -30, ay: -30 });
                                plotProfile(true);
                            } else if (self.markerAction() === "targetPoint") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Target'));
                                self.target_position = clickedZ;
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'Target: '+self.target_position, showarrow: true, arrowhead: 2, ax: 20, ay: 20 });
                                plotProfile(true);
                            } else if (self.markerAction() === "refset") {
                                var offset = getSmartAnnotationOffset(clickedX, clickedZ, self.xValues, self.zValues);
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('D'));
                                self.referenceZ = clickedX;
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'D='+self.refdiam(), showarrow: true, arrowhead: 2, ax: offset.ax, ay: offset.ay });
                                plotProfile(false);
                            }
                        } else if (self.isXFile) {
                            if (self.markerAction() === "Max") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Max'));
                                if (self.vMin && clickedX < self.vMin) {
                                    alert("Max must be greater than Min");
                                    return;
                                }
                                self.vMax = clickedX;
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'Max: '+self.vMax, showarrow: true, arrowhead: 2, ax: 30, ay: -30 });
                                if (!self.do_distance()) { plotProfile(false); }
                            } else if (self.markerAction() === "Min") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Min'));
                                if (self.vMax && clickedX > self.vMax) {
                                    alert("Min must be less than Max");
                                    return;
                                }
                                self.vMin = clickedX;
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'Min: '+self.vMin, showarrow: true, arrowhead: 2, ax: -30, ay: -30 });
                                if (!self.do_distance()) { plotProfile(false); }
                            } else if (self.markerAction() === "targetPoint") {
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('Target'));
                                self.target_position = clickedX;
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'Target: '+self.target_position, showarrow: true, arrowhead: 2, ax: 20, ay: 20 });
                                plotProfile(false);
                            } else if (self.markerAction() === "refset") {
                                var offset = getSmartAnnotationOffset(clickedX, clickedZ, self.xValues, self.zValues);
                                self.annotations = self.annotations.filter(a => !a.text.startsWith('D'));
                                self.referenceZ = clickedZ;
                                self.annotations.push({ x: clickedX, y: clickedZ, xref: 'x', yref: 'y', text: 'D='+self.refdiam(), showarrow: true, arrowhead: 2, ax: offset.ax, ay: offset.ay });
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

        $("#svgFileSelect").on("change", function () {
            var filePath = $("#svgFileSelect option:selected").attr("download");
            if (!filePath) {
                self.selectedSVGFile = null;
                return;
            }
            var theindex = $("#svgFileSelect option:selected").attr("index");
            self.selectedSVGFile = self.svgfiles[theindex];
        });

        // ── Polar editor ──────────────────────────────────────────────────────
        self.openPolarEditor = function () {
            var segments = self.segments();
            var angles = [];
            var step = 360.0 / segments;
            for (var i = 0; i < segments; i++) angles.push(step * i);
            self.polarAngles(angles);
            drawPolarPlot();
            $("#polarEditorModal").show();
        };

        function drawPolarPlot() {
            var angles = self.polarAngles();
            var n = self.laser_sections().length;

            // Working layer outer edge extends to meet the innermost committed shell
            // (or stays at WORK_R when there are no committed layers yet)
            var n = self.laser_sections().length;
            var workOuter = WORK_R + (MAX_SHELLS - n) * SHELL_W;
            // Current working layer — from 0.25 up to the innermost committed shell
            var traces = angles.map(function(a) {
                return {
                    type: "scatterpolar",
                    r: [0.25, workOuter],
                    theta: [a, a],
                    mode: "lines",
                    line: { color: "black", width: 1.5 },
                    showlegend: false,
                    hoverinfo: "skip"
                };
            });

            // Committed layers — static outer shells, first committed = outermost
            // Layer i=0 → r_outer = WORK_R + MAX_SHELLS * SHELL_W = 2.0
            //              r_inner = WORK_R + (MAX_SHELLS - 1) * SHELL_W = 1.9
            // Layer i=1 → r_outer = 1.9, r_inner = 1.8  … and so on inward
            self.laser_sections().forEach(function(layer, i) {
                var r_outer = WORK_R + (MAX_SHELLS - i) * SHELL_W;
                var r_inner = WORK_R + (MAX_SHELLS - i - 1) * SHELL_W;
                layer.angles.forEach(function(a) {
                    traces.push({
                        type: "scatterpolar",
                        r: [r_inner, r_outer],
                        theta: [a, a],
                        mode: "lines",
                        line: { color: layer.color, width: 2 },
                        showlegend: false,
                        hoverinfo: "skip"
                    });
                });
            });

            var layout = {
                polar: {
                    radialaxis: {
                        visible: false,
                        range: [0, WORK_R + MAX_SHELLS * SHELL_W],  // static [0, 2.0]
                        showgrid: false,
                        showline: false
                    },
                    angularaxis: { direction: "clockwise", rotation: 90, showgrid: false, showline: false }
                },
                showlegend: false,
                margin: { t: 10, b: 10, l: 10, r: 10 }
            };

            Plotly.newPlot("polarPlot", traces, layout).then(function(gd) {
                gd.removeAllListeners && gd.removeAllListeners("click");
                gd.onclick = function(evt) {
                    var theta = pixelToPolarTheta(gd, evt);
                    if (theta != null) {
                        handlePolarClick(theta);
                    }
                };
            });
        }

        function pixelToPolarTheta(gd, evt) {
            var fullLayout = gd._fullLayout;
            var polar = fullLayout.polar;
            if (!polar) return null;

            var rect = gd.getBoundingClientRect();
            var clickX = evt.clientX - rect.left;
            var clickY = evt.clientY - rect.top;

            var xDomain = polar.domain.x;
            var yDomain = polar.domain.y;
            var plotWidth = fullLayout.width;
            var plotHeight = fullLayout.height;

            var px0 = xDomain[0] * plotWidth;
            var px1 = xDomain[1] * plotWidth;
            var py0 = (1 - yDomain[1]) * plotHeight;
            var py1 = (1 - yDomain[0]) * plotHeight;

            var cx = (px0 + px1) / 2;
            var cy = (py0 + py1) / 2;

            var dx = clickX - cx;
            var dy = clickY - cy;

            var mathAngle = Math.atan2(-dy, dx) * 180 / Math.PI;
            mathAngle = (mathAngle + 360) % 360;

            var rotation = 90;
            var theta = (rotation - mathAngle + 360) % 360;
            return theta;
        }

        function angleDelta(a, b) {
            var d = Math.abs(a - b) % 360;
            return d > 180 ? 360 - d : d;
        }

        function nearestNeighbors(clickAngle, angles) {
            var sorted = angles.slice().sort(function(a, b) { return a - b; });
            var lower = null, upper = null;
            for (var i = 0; i < sorted.length; i++) {
                var next = sorted[(i + 1) % sorted.length];
                var span = (next - sorted[i] + 360) % 360;
                var offset = (clickAngle - sorted[i] + 360) % 360;
                if (offset <= span) {
                    lower = sorted[i];
                    upper = next;
                    break;
                }
            }
            return [lower, upper];
        }

        function applySymmetry(newAngle, isAdd) {
            var sym = parseInt(self.polarSymmetry());
            var step = 360 / sym;
            var angles = self.polarAngles();
            var updated = angles.slice();
            for (var k = 0; k < sym; k++) {
                var a = (newAngle + step * k) % 360;
                if (isAdd) {
                    if (!updated.some(function(x) { return angleDelta(x, a) < 0.01; })) {
                        updated.push(a);
                    }
                } else {
                    updated = updated.filter(function(x) { return angleDelta(x, a) >= 0.01; });
                }
            }
            self.polarAngles(updated);
        }

        function handlePolarClick(clickTheta) {
            var mode = $("input[name='polarMode']:checked").val();
            var angles = self.polarAngles();

            if (mode === "add") {
                var neighbors = nearestNeighbors(clickTheta, angles);
                if (neighbors[0] == null) return;
                var mid = (neighbors[0] + (((neighbors[1] - neighbors[0]) + 360) % 360) / 2) % 360;
                applySymmetry(mid, true);
            } else {
                var closest = angles.reduce(function(best, a) {
                    var d = angleDelta(a, clickTheta);
                    return (d < best.d) ? { a: a, d: d } : best;
                }, { a: null, d: Infinity });
                if (closest.a != null && closest.d < 10) {
                    applySymmetry(closest.a, false);
                }
            }
            drawPolarPlot();
        }

        // ── Layer management ──────────────────────────────────────────────────

        /**
         * Commit the current Min/Max range + polar line layout as a new layer.
         * Draws a colored band on the profile plot and adds the lines to the
         * inner ring of the polar plot.
         */
        self.commitLayer = function() {
            if (self.vMin === null || self.vMin === undefined ||
                self.vMax === null || self.vMax === undefined) {
                alert("Set Min and Max before committing a layer.");
                return;
            }
            if (self.polarAngles().length === 0) {
                alert("Open the polar editor and define at least one line before committing.");
                return;
            }

            var layerIndex = self.laser_sections().length;
            var color = LAYER_COLORS[layerIndex % LAYER_COLORS.length];

            self.laser_sections.push({
                id: layerIndex + 1,
                vMin: self.vMin,
                vMax: self.vMax,
                angles: self.polarAngles().slice(),   // snapshot
                color: color
            });

            // Clear Min/Max annotations and working values for the next layer
            self.vMin = null;
            self.vMax = null;
            self.annotations = self.annotations.filter(
                a => !a.text.startsWith('Min') && !a.text.startsWith('Max')
            );

            plotProfile(self.isZFile);

            // Refresh polar plot if it's currently visible
            if ($("#polarEditorModal").is(":visible")) {
                drawPolarPlot();
            }
        };

        /** Remove a single committed layer (called from the KO foreach in the template). */
        self.removeLayer = function(layer) {
            self.laser_sections.remove(layer);
            // Re-number the remaining layers
            self.laser_sections().forEach(function(l, i) { l.id = i + 1; });
            // Force KO to notify (objects are mutated, not replaced)
            self.laser_sections.valueHasMutated();
            plotProfile(self.isZFile);
            if ($("#polarEditorModal").is(":visible")) {
                drawPolarPlot();
            }
        };

        /** Remove all committed layers after confirmation. */
        self.clearAllLayers = function() {
            if (self.laser_sections().length === 0) return;
            if (confirm("Remove all " + self.laser_sections().length + " committed layer(s)?")) {
                self.laser_sections([]);
                plotProfile(self.isZFile);
                if ($("#polarEditorModal").is(":visible")) {
                    drawPolarPlot();
                }
            }
        };

        // ── File / scan events ────────────────────────────────────────────────
        $("#scan_file_select").on("change", function () {
            var filePath = $("#scan_file_select option:selected").attr("path");
            self.name = $("#scan_file_select option:selected").attr("value");
            if (!filePath) return;

            self.isZFile = $("#scan_file_select option:selected").text().startsWith("Z");
            self.isXFile = $("#scan_file_select option:selected").text().startsWith("X");
            if (!$("#scan_file_select option:selected").text().endsWith("txt")) {
                alert("Selected file is not a text scan file.");
                return;
            }

            if (self.isZFile) { $(".zscan").show(); }
            else              { $(".zscan").hide(); }

            self.annotations = [];
            self.vMax = null;
            self.vMin = null;
            self.target_position = null;

            self.createGraph(filePath);
        });

        $("#reverseZButton").on("click", function () {
            if (self.isZFile) {
                self.zValues.reverse();
                plotProfile(true);
            } else {
                self.xValues.reverse();
                plotProfile(false);
            }
        });

        self.getPointsInRange = function() {
            var pointsInRange = [];
            for (var i = 0; i < self.xValues.length; i++) {
                var x = parseFloat(self.xValues[i]);
                var z = parseFloat(self.zValues[i]);
                var rangeVal = self.isZFile ? z : x;
                if (self.vMin != null && rangeVal < self.vMin) continue;
                if (self.vMax != null && rangeVal > self.vMax) continue;
                pointsInRange.push({ x: x.toFixed(3), z: z.toFixed(3) });
            }
            return pointsInRange;
        };

        self.onDataUpdaterPluginMessage = function(plugin, data) {
            if (plugin == 'Profiler' && data.laser === false || data.laser === true) {
                self.laser_mode(data.laser);
            }

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
                    x: 0.5, y: 1,
                    xref: 'paper', yref: 'paper',
                    text: 'Width: '+self.width+'<br>Pro. Dist.: '+self.pd,
                    showarrow: false,
                });
                plotProfile(self.isZFile);
            }

            if (plugin == 'Profiler' && data.type === "polar_angles") {
                self.polarAngles(data.angles);
                drawPolarPlot();
            }
        };

        self.createGraph = function(filePath) {
            OctoPrint.simpleApiCommand("profiler", "creategraph", { filepath: filePath })
                .done(function() { console.log("Graph info transmitted"); })
                .fail(function() { console.error("Graph info not transmitted"); });
        };

        self.send_error_messasge = function(message) {
            OctoPrint.simpleApiCommand("latheengraver", "send_error_message", { message: message })
                .done(function() { console.log("Error message sent"); })
                .fail(function() { console.error("Error message not sent"); });
        };

        self.get_pd = function() {
            OctoPrint.simpleApiCommand("profiler", "get_arc_length", { vMin: self.vMin, vMax: self.vMax })
                .done(function() { console.log("Info for arc length sent"); })
                .fail(function() { console.error("Did not get arc length"); });
        };

        // ── Write Job ─────────────────────────────────────────────────────────
        self.writeGCode = function() {
            if (self.mode() == "none") {
                alert("Mode must be set to write a job.");
                return;
            }

            // Laser mode: require at least one committed layer
            if (self.mode() === "laser") {
                if (self.laser_sections().length === 0) {
                    alert("Commit at least one layer before writing a laser job.");
                    return;
                }
            } else {
                if (self.vMax === null || self.vMax === undefined ||
                    self.vMin === null || self.vMin === undefined) {
                    alert("Min. and Max. values must be set.");
                    return;
                }
            }

            if (self.mode() == "facet" || self.mode() == "wrap" || self.mode() == "flute") {
                if (self.referenceZ === null || Number(self.diam) < 1) {
                    alert("Reference Diameter must be set.");
                    return;
                }
            }

            if (self.mode() == "flute") {
                if (Number(self.step_down()) > Number(self.depth()) || Number(self.step_down()) <= 0) {
                    alert("Step down must be less than or equal to total depth and greater than 0.");
                    return;
                }
            }

            if (Number(self.tool_length()) < 10) {
                alert("You must provide rotation center to surface distance");
                return;
            }

            var clearance;
            if (self.isZFile) { clearance = Math.max(...self.xValues); }
            else              { clearance = Math.max(...self.zValues); }

            var plot = self.getPointsInRange();

            // Serialize committed laser layers for the backend
            var serializedLayers = self.laser_sections().map(function(layer) {
                return {
                    id:     layer.id,
                    vMin:   layer.vMin,
                    vMax:   layer.vMax,
                    angles: layer.angles,
                    color:  layer.color
                };
            });

            var data = {
                plot_data: plot,
                mode: self.mode(),
                tool_length: self.tool_length(),
                max_B: self.max_B(),
                min_B: self.min_B(),
                power: self.power(),
                //angles: self.polarAngles(),          // current working angles (single-layer fallback)
                feed: self.feed(),
                test: self.test(),
                segments: self.segments(),
                vMax: self.vMax,
                vMin: self.vMin,
                laser_sections: serializedLayers,    // ← multi-layer payload
                filename: self.selectedGCodeFile,
                svgfile: self.selectedSVGFile,
                diam: self.refdiam(),
                clear: clearance,
                risky: self.risky(),
                refZ: self.referenceZ,
                arotate: self.Arot(),
                side: self.side(),
                name: self.name,
                depth: self.depth(),
                step_down: self.step_down(),
                leadin: self.leadin(),
                leadout: self.leadout(),
                width: self.width,
                radius_adjust: self.radius_adjust(),
                singleB: self.singleB(),
                steps: self.increment(),
                smoothing: self.smoothing(),
                step_over: self.step_over(),
                tool_diam: self.tool_diam(),
                facet_invert: self.facet_invert(),
                depth_mod: self.depth_mod(),
                adaptive: self.adaptive(),
                feedscale: self.feedscale(),
                ignore_oval: self.ignore_oval(),
                conventional: self.conventional(),
                extra_depth: self.extra_depth(),
                flute_gap: self.flute_gap(),
            };

            OctoPrint.simpleApiCommand("profiler", "write_job", data)
                .done(function() { console.log("GCode written successfully."); })
                .fail(function() { console.error("Failed to write GCode."); });
        };

        self.gotoposition = function(getB) {
            if (self.isZFile && self.side == "none") {
                alert("Tool direction must be set for Z scans");
                return;
            }
            if (self.tool_length() < 10.0) {
                alert("You must provide rotation center to surface distance");
                return;
            }

            var clearance;
            if (self.isZFile) {
                clearance = (self.side === "back")
                    ? Math.abs(Math.min(...self.xValues))
                    : Math.max(...self.xValues);
            } else {
                clearance = Math.max(...self.zValues);
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
                smoothing: self.smoothing(),
                getB: getB,
            };
            OctoPrint.simpleApiCommand("profiler", "go_to_position", data)
                .done(function() { console.log("Go to target successful."); })
                .fail(function() { console.error("Failed to go to target"); });
        };

        self.onTabChange = function(current, previous) {
            if (current === "#tab_plugin_profiler") {
                self.fetchProfileFiles();
                self.fetchWrapFiles();
                self.fetchsvgFiles();
            }
        };
    }

    OCTOPRINT_VIEWMODELS.push({
        construct: ProfilerViewModel,
        dependencies: ["loginStateViewModel", "settingsViewModel"],
        elements: ["#tab_plugin_profiler"],
        onTabChange: true
    });
});

function getSmartAnnotationOffset(x, y, xArray, yArray) {
    var xMin = Math.min(...xArray);
    var xMax = Math.max(...xArray);
    var yMin = Math.min(...yArray);
    var yMax = Math.max(...yArray);
    var xMid = (xMin + xMax) / 2;
    var yMid = (yMin + yMax) / 2;
    return {
        ax: x < xMid ?  30 : -30,
        ay: y < yMid ?  30 : -30
    };
}

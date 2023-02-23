import networkx as nx
import numpy as np
import yaml

class PhotonicCircuit():
    def __init__(self,netlist, models_dict, source=None, destinations=None):
        self.netlist = netlist
        self.models_dict = models_dict
        self.source = source 
        self.destinations = destinations
        self.construct_graph_from_netlist(self.netlist,self.models_dict)

    def construct_graph_from_netlist(self,netlist,models_dict):
        edges = netlist['connections']
        nodes = netlist['instances']
        positions = netlist['placements']
        self.overall_ports = netlist['ports']
        self.overall_name = netlist['name']

        G = nx.Graph() #Create a directed graph. Note, this does not simulate the effects of Fabry Perot yet
        for i in nodes.keys():
            try:
                try:
                    info = nodes[i]['info']
                except:
                    info = {}
                try:
                    settings = nodes[i]['settings']
                except:
                    settings = {}
                G.add_node(
                    i,
                    name=str(i), 
                    model=models_dict[nodes[i]['component']](info=info,settings=settings),
                    ready=False,
                    )
                
                G.nodes[i]['port_names'] = G.nodes[i]['model'].port_names
            except Exception as e:
                print(f"Model for component {e} not found")

        for i in edges.keys():
            source = i.split(",")[0]
            destination = edges[i].split(",")[0]
            name = i.split(",")[0]
            G.add_edge(
                source,
                destination,
                port_mapping = {
                    source: i.split(",")[1],
                    destination: edges[i].split(",")[1],
                },
                name=name,
                forward_propagating_field_value=None,
                # backward_propagating_field_value=None #TODO: Add in Fabry-perot style effects
            ) #Initialize graph to have "None"=no value for the propagating fields

        # Add in nodes for all of the io ports and connect them to their respective components
        io_optical_ports = []
        for i in self.overall_ports:
            if i[0] == 'o':
                io_optical_ports.append(self.overall_ports[i])
                
        self.graph, self.positions, self.ports =  G, positions, io_optical_ports

    def add_laser_port(self,port_to_connect_laser_to, positions):
        # This function adds a "laser/source" to the circuit
        # Source port is the port and component we are connecting to in a format like "cp1,o1"
        self.graph.add_node(
            "input_laser",
            name="input_laser",
            model=self.models_dict['laser'](),
            ready=True
        )

        node1 = 'input_laser'
        node2 = port_to_connect_laser_to.split(",")[0]

        self.graph.add_edge(
            node1,
            node2,
            port_mapping = {
                node1: "o1",
                node2: port_to_connect_laser_to.split(",")[1],
            },
            name="input_laser",
            forward_propagating_field_value=1, #Laser power is "1"
        )

        #Update the position of the laser
        input_laser_dict = {
            "x":positions[node2]['x'] - 80,
            "y":positions[node2]['y'],
            "rotation":positions[node2]['rotation'],
            "mirror":positions[node2]['mirror'],
        }
        positions['input_laser'] = input_laser_dict

    def add_photodetector_port(self,port_to_connect_photodetector_to, positions):
        self.graph.add_node(
            "output_photodetector",
            name="output_photodetector",
            model=self.models_dict['detector'](),
            ready=True
        )

        node1 = port_to_connect_photodetector_to.split(",")[0]
        node2 = 'output_photodetector'

        self.graph.add_edge(
            node1,
            node2,
            port_mapping = {
                node1: port_to_connect_photodetector_to.split(",")[1],
                node2: "o1",
            },
            name="output_photodetector",
            forward_propagating_field_value=None, #Laser power is "1"
        )

        # Update the position of the detector
        output_photodetector_dict = {
            "x":positions[node1]['x'] + 80,
            "y":positions[node1]['y'],
            "rotation":positions[node1]['rotation'],
            "mirror":positions[node1]['mirror'],
        }
        positions['output_photodetector'] = output_photodetector_dict

    def add_fielddetector_port(self,port_to_connect_fielddetector_to, positions):
        self.graph.add_node(
            "output_fielddetector",
            name="output_fielddetector",
            model=self.models_dict['detector'](),
            ready=True
        )

        node1 = port_to_connect_fielddetector_to.split(",")[0]
        node2 = 'output_fielddetector'

        self.graph.add_edge(
            node1,
            node2,
            port_mapping = {
                node1: port_to_connect_fielddetector_to.split(",")[1],
                node2: "o1",
            },
            name="output_fielddetector",
            forward_propagating_field_value=None, #Laser power is "1"
        )

        # Update the position of the detector
        output_fielddetector_dict = {
            "x":positions[node1]['x'] + 80,
            "y":positions[node1]['y'],
            "rotation":positions[node1]['rotation'],
            "mirror":positions[node1]['mirror'],
        }
        positions['output_fielddetector'] = output_fielddetector_dict

    def reset_edge_values(self):
        for e in self.graph.edges:
            if ("laser" in e[0]) or ("laser" in e[1]):
                pass
            else:
                self.graph.edges[e]['forward_propagating_field_value'] = None
    
    def set_all_nodes_ready_false(self):
        for n in self.graph.nodes:
            if ("laser" not in self.graph.nodes[n]['name']) and ("detector" not in self.graph.nodes[n]['name']):
                self.graph.nodes[n]['ready'] = False

    def are_all_nodes_ready(self):
        bool_list = []
        for n in self.graph.nodes:
            bool_list.append(self.graph.nodes[n]['ready'])
        return all(bool_list)

    def get_values_from_attached_edges(self,node):
        # Returns a dictionary with elements in the form (port: value) where
        # port is the port corresponding to node
        returnable = {}
        for i in self.graph.edges(node):
            temp = self.graph.edges[i]["port_mapping"][node]
            returnable[temp] = self.graph.edges[i]['forward_propagating_field_value']
        return returnable

    def add_value_to_attached_edge(self,node,port,value):
        for i in self.graph.edges(node):
            if self.graph.edges[i]['port_mapping'][node] == port:
                self.graph.edges[i]['forward_propagating_field_value'] = value

    def are_all_values_in_edges_floats(self,node):
        storage = []
        for e in self.graph.edges(node):
            storage.append(self.graph.edges[e]['forward_propagating_field_value'])
        return any(storage)

    def propagate_field(self,node):
        # This function looks at a node and see's if it can propagate the field forward
        if ("laser" in node) or ("detector" in node):
            return None
        values_from_attached_edges = self.get_values_from_attached_edges(node)
        # print(node, values_from_attached_edges)
        s_matrix = self.graph.nodes[node]['model']()
        # Forward propagation step
        #Check that all values are available in from_ports
        from_port_storage = []
        for element in s_matrix:
            from_port, to_port = element
            from_port_storage.append(values_from_attached_edges[from_port])

        values_to_write = {}
        port_names = self.graph.nodes[node]['port_names']
        for p in port_names:
            values_to_write[p] = None
        for element in s_matrix:
            from_port, to_port = element
            try:
                from_port_value = values_from_attached_edges[from_port]
                value_to_write = s_matrix[element]*from_port_value
                if values_to_write[to_port] == None:
                    values_to_write[to_port] = 0
                values_to_write[to_port] += value_to_write
            except:
                pass
        
        for p in port_names:
            if values_to_write[p] != None:
                self.add_value_to_attached_edge(node,p,values_to_write[p])

        if self.are_all_values_in_edges_floats(node):
            self.graph.nodes[node]['ready'] = True

    def return_nodes_not_ready(self):
        return [x for x,y in self.graph.nodes(data=True) if y['ready']==False]

    def update_all_edges(self):
        # First, set all edges "ready" status to False
        self.set_all_nodes_ready_false()
        # Reset all edge values except the laser
        self.reset_edge_values()
        # Iterate over all nodes/components in the network, see if they are able to propagate the light
        # Keep a counter so that we only iterate over the graph a reasonable number of times
        max_number_iterations = 1000
        while (self.are_all_nodes_ready() == False) and (max_number_iterations > 0):
            #TODO: Have this iterate over the graph using bredth first search
            for n in self.return_nodes_not_ready():
                self.propagate_field(n)
            max_number_iterations -= 1
            if max_number_iterations == 0:
                print("Ran out of iterations while updating all edges")
        self.propagate_field(n)

    def update_wavelength(self,wl):
        for n in self.graph.nodes:
            self.graph.nodes[n]['model'].update_wavelength(wl=wl)

    def update_time(self):
        for n in self.graph.nodes:
            self.graph.nodes[n]['model'].update_time()

    def readout_from_photodetectors(self):
        #Find detector nodes
        detectors = []
        for n in self.graph.nodes:
            if "photodetector" in self.graph.nodes[n]['name']:
                detectors.append(n)

        detector_values = []
        for d in detectors:
            temp = self.get_values_from_attached_edges(d)['o1']
            detector_values.append(temp)
        detector_values = np.array(detector_values)
        return np.real(detector_values*np.conj(detector_values))

    def readout_from_fielddetectors(self):
        #Find detector nodes
        fielddetectors = []
        for n in self.graph.nodes:
            if "fielddetector" in self.graph.nodes[n]['name']:
                fielddetectors.append(n)

        detector_values = []
        for d in fielddetectors:
            temp = self.get_values_from_attached_edges(d)['o1']
            detector_values.append(temp)
        return np.array(detector_values)

    def return_components_callback(self,components):
        returnable = []
        for c in components:
            returnable.append(self.graph.nodes[c]['model'])
        return returnable
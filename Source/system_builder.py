import Source.FastComposedModels_pb2 as fcm
import copy
import warnings


class SystemBuilder:

	def __check_existence(self, c):
		if c.id in self.components:
			print(self.system)
		assert c.id not in self.components, "Two components with the same id: " + c.id

	def __init__(self, verbose=True, id=""):
		self.components = {}
		self.system = fcm.System()
		self.verbose = verbose
		self.metrics = {}
		self.start = ""  # Indicates the beginning of the graph/ensemble
		self.id = id

	def add_data(self, data):
		self.__check_existence(data)
		self.system.data.extend([data])
		self.components[data.id] = self.system.data[-1]
		if self.verbose:
			print("Data component {} added to the system".format(data.id))

	def add_classifier(self, classifier, trigger_ids=None, merger_ids=None):
		# TODO: If trigger_ids not null, update triggers and connect to classifier
		self.__check_existence(classifier)

		"""
		if classifier.HasField('component_id'):
			assert classifier.component_id in self.components, \
				"Error in Classifier: " + classifier.component_id + "in classifier should exist in the system"
		"""

		self.system.classifier.extend([classifier])
		self.components[classifier.id] = self.system.classifier[-1]
		if self.verbose:
			print("Classifier component {} added to the system".format(classifier.id))

	def add_trigger(self, trigger, trigger_ids=None):
		# TODO: If trigger_ids not null, update triggers and connect to classifier
		self.__check_existence(trigger)

		"""
		for id in trigger.component_ids:
			assert id in self.components, \
				"ERROR in Trigger: component with id " + id + " not in the system"
		"""

		self.system.trigger.extend([trigger])
		self.components[trigger.id] = self.system.trigger[-1]
		if self.verbose:
			print("Trigger component {} added to the system".format(trigger.id))

	def add_merger(self, merger):
		self.__check_existence(merger)
		assert len(merger.component_ids) > 1, \
			"ERROR in Merger" + id + ": Merger should have at least two classifiers"
		for id in merger.component_ids:
			assert id in self.components, \
				"ERROR in Merger: component with id " + id + " not in the system"
			assert self.components[id].DESCRIPTOR.name == "Classifier", \
				"ERROR in Merger: Merger should merge only classifiers"
		self.system.merger.extend([merger])
		self.components[merger.id] = self.system.merger[-1]
		if self.verbose:
			print("Trigger component {} added to the system".format(merger.id))

	# Removes a component from the system
	def remove(self, id):
		# TODO: Remove connections to triggers in the system
		if id not in self.components:
			return
		type_component = self.components[id].DESCRIPTOR.name
		del self.components[id]

		if type_component == "Trigger":
			for i, c in enumerate(self.system.trigger):
				if c.id == id: del self.system.trigger[i]
		elif type_component == "Classifier":
			for i, c in enumerate(self.system.classifier):
				if c.id == id: del self.system.classifier[i]
		elif type_component == "Merger":
			for i, c in enumerate(self.system.merger):
				if c.id == id: del self.system.merger[i]
		elif type_component == "Data":
			for i, c in enumerate(self.system.data):
				if c.id == id: del self.system.data[i]

		if self.verbose:
			print("Component {} removed from the system".format(id))

	# Updates an existing component
	def replace(self, id, new_component):
		# TODO: Update connections to triggers!!!
		if self.verbose:
			print("Replacing component {} by {}".format(id, new_component.id))
		self.remove(id)
		if new_component.DESCRIPTOR.name == "Trigger":
			self.add_trigger(new_component)
		elif new_component.DESCRIPTOR.name == "Classifier":
			self.add_classifier(new_component)
		elif new_component.DESCRIPTOR.name == "Merger":
			self.add_merger(new_component)
		elif new_component.DESCRIPTOR.name == "Data":
			self.add_data(new_component)
		else:
			raise Exception("Component Type not recognized")
		self.components[new_component.id].CopyFrom(new_component)

	def build_classifier_dict(self, name, start_id, phases=["test"]):
		import Source.system_evaluator as eval
		classifier_dict = {'name': name, 'test':{}, 'train':{}, 'val':{}}
		eval_results = eval.evaluate(self, start_id, classifier_dict=classifier_dict, phases=phases)
		return classifier_dict, eval_results

	# Creates a copy of the ensemble system
	def copy(self):
		new_sys = SystemBuilder()
		new_sys.system = copy.deepcopy(self.system)
		new_sys.verbose = self.verbose
		new_sys.start = self.start
		new_sys.components = {}
		for data in new_sys.system.data:
			new_sys.components[data.id] = data
		for trigger in new_sys.system.trigger:
			new_sys.components[trigger.id] = trigger
		for classifier in new_sys.system.classifier:
			new_sys.components[classifier.id] = classifier
		for merger in new_sys.system.merger:
			new_sys.components[merger.id] = merger
		return new_sys

	# ------------- GETTERS AND SETTERS --------------------#
	# Sets the start of the graph for evaluation
	def set_start(self, id):
		if id not in self.components:
			warnings.warn("WARNING: %s not in the system" % id)
		self.start = id

	# Sets system id
	def set_sysid(self, id):
		self.id = id

	# Returns the component in protobuf message
	def get(self, id):
		if id not in self.components:
			print(self.system)
		return self.components[id]

	# Get start component for evaluation
	def get_start(self):
		return self.start

	# Get the protobuf definition (graph) of the ensemble system
	def get_message(self):
		return self.system

	# Get system's id
	def get_sysid(self):
		return self.id

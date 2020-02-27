import Source.FastComposedModels_pb2 as fcm
import copy
import warnings


class SystemBuilder:

	def __check_existence(self, c):
		exists = self.get(c.id) is not None
		if exists:
			print(self.get_sysid())
			print(self.get_message())
		assert not exists, "Two components with the same id: " + c.id

	def __deserialize(self):
		system = fcm.System()
		system.ParseFromString(self.system_serialized)
		return system

	def __serialize(self, system):
		self.system_serialized = system.SerializeToString()

	def __init__(self, verbose=True, id=""):
		self.system_serialized = fcm.System().SerializeToString()
		self.verbose = verbose
		self.metrics = {}
		self.start = ""  # Indicates the beginning of the graph/ensemble
		self.id = id

	def add_data(self, data):
		self.__check_existence(data)
		system = self.__deserialize()
		system.data.extend([data])
		self.__serialize(system)
		if self.verbose:
			print("Data component {} added to the system".format(data.id))

	def add_classifier(self, classifier):
		self.__check_existence(classifier)
		system = self.__deserialize()
		system.classifier.extend([classifier])
		self.__serialize(system)
		if self.verbose:
			print("Classifier component {} added to the system".format(classifier.id))

	def add_trigger(self, trigger, trigger_ids=None):
		self.__check_existence(trigger)
		system = self.__deserialize()
		system.trigger.extend([trigger])
		self.__serialize(system)
		if self.verbose:
			print("Trigger component {} added to the system".format(trigger.id))

	def add_merger(self, merger):
		self.__check_existence(merger)
		assert len(merger.component_ids) > 1, \
			"ERROR in Merger" + merger.id + ": Merger should have at least two classifiers"
		for id in merger.component_ids:
			assert self.get(id) is not None, \
				"ERROR in Merger: component with id " + id + " not in the system"
			assert self.get(id).DESCRIPTOR.name == "Classifier", \
				"ERROR in Merger: Merger should merge only classifiers"
		system = self.__deserialize()
		system.merger.extend([merger])
		self.__serialize(system)
		if self.verbose:
			print("Trigger component {} added to the system".format(merger.id))

	# Removes a component from the system
	def remove(self, id):
		if self.get(id) is None: return

		type_component = self.get(id).DESCRIPTOR.name
		system = self.__deserialize()
		if type_component == "Trigger":
			for i, c in enumerate(system.trigger):
				if c.id == id: del system.trigger[i]
		elif type_component == "Classifier":
			for i, c in enumerate(system.classifier):
				if c.id == id: del system.classifier[i]
		elif type_component == "Merger":
			for i, c in enumerate(system.merger):
				if c.id == id: del system.merger[i]
		elif type_component == "Data":
			for i, c in enumerate(system.data):
				if c.id == id: del system.data[i]
		self.__serialize(system)

		if self.verbose:
			print("Component {} removed from the system".format(id))

	# Updates an existing component
	def replace(self, id, new_component):
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
		# self.components[new_component.id].CopyFrom(new_component)

	def build_classifier_dict(self, name, start_id, phases=["test"]):
		import Source.system_evaluator as eval
		classifier_dict = {'name': name, 'test':{}, 'train':{}, 'val':{}}
		eval_results = eval.evaluate(self, start_id, classifier_dict=classifier_dict, phases=phases)
		return classifier_dict, eval_results

	# Creates a copy of the ensemble system
	def copy(self):
		new_sys = SystemBuilder()
		new_sys.system_serialized = copy.copy(self.system_serialized)
		new_sys.verbose = copy.copy(self.verbose)
		new_sys.start = copy.copy(self.start)
		return new_sys

	# ------------- GETTERS AND SETTERS --------------------#
	# Sets the start of the graph for evaluation
	def set_start(self, id):
		self.start = id

	# Sets system id
	def set_sysid(self, id):
		self.id = id

	# Returns the component in protobuf message
	def get(self, id):
		system = self.__deserialize()
		for descriptor_field in system.DESCRIPTOR.fields:
			components = getattr(system, descriptor_field.name)
			for component in components:
				if component.id == id:
					return component
		return None

	# Get start component for evaluation
	def get_start(self):
		return self.start

	# Get the protobuf definition (graph) of the ensemble system
	def get_message(self):
		system = self.__deserialize()
		return system

	# Get system's id
	def get_sysid(self):
		return self.id

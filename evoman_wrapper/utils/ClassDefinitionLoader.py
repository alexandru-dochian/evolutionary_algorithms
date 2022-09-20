import importlib


class ClassDefinitionLoader:
    EA_PACKAGE_NAME = "evoman_wrapper.implementations"
    CONTROLLER_PACKAGE_NAME = "evoman_wrapper.controllers"

    @staticmethod
    def get_evolutionary_algorithm_instance(class_name):
        return ClassDefinitionLoader.__get_class_definition(ClassDefinitionLoader.EA_PACKAGE_NAME, class_name)
        pass

    @staticmethod
    def get_controller_instance(class_name):
        return ClassDefinitionLoader.__get_class_definition(ClassDefinitionLoader.CONTROLLER_PACKAGE_NAME, class_name)

    @staticmethod
    def __get_class_definition(package_name, class_name):
        return getattr(importlib.import_module(
            ClassDefinitionLoader.__get_module_name(package_name, class_name)),
            class_name
        )

    @staticmethod
    def __get_module_name(package_name, class_name):
        return ".".join([package_name, class_name])

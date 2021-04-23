import torchvision

from Logic.ProperLogic.handlers.handler_edit_labels import edit_labels
from Logic.ProperLogic.handlers.handler_view_person import view_person
from Logic.ProperLogic.handlers.handler_process_image_dir import process_image_dir
from Logic.ProperLogic.handlers.handler_reclassify import reclassify
from Logic.ProperLogic.handlers.handler_reset_cluster_ids import reset_cluster_ids
from Logic.ProperLogic.handlers.handler_clear_data import clear_data
from Logic.ProperLogic.misc_helpers import log_error, have_equal_type_names

# TODO: Where to put this and how to handle general case?
IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'

# TODO: Where to put these?
TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()


# INPUT_SIZE = [112, 112]


class Command:
    # TODO: add 'help' command
    commands = dict()

    def __init__(self, cmd_name, cmd_desc, cmd_shorthand, handler=None, handler_params=None):
        if not cmd_shorthand:
            raise ValueError('command shorthand cannot be empty')
        elif cmd_shorthand not in cmd_name:
            raise ValueError(f"command name '{cmd_name}' doesn't start with '{cmd_shorthand}'")
        elif cmd_shorthand in self.get_command_shorthands():
            # TODO: Also output to which command?
            raise ValueError(f"command shorthand '{cmd_shorthand}' of new command '{cmd_name}' is already assigned to"
                             " a different command")

        if handler_params is None:
            handler_params = []
        self.cmd_name = cmd_name
        self.cmd_desc = cmd_desc
        self.cmd_shorthand = cmd_shorthand
        self.handler = handler
        self.handler_params = handler_params
        type(self).commands[self.cmd_name] = self

    def __eq__(self, other):
        # TODO: Implement more strict checking?
        if not have_equal_type_names(self, other):
            return False
        return self.cmd_name == other.cmd_name

    def __str__(self):
        return self.cmd_name

    def get_cmd_name(self):
        return self.cmd_name

    def set_cmd_name(self, cmd_name):
        self.cmd_name = cmd_name

    def get_cmd_description(self):
        return self.cmd_desc

    def set_cmd_description(self, new_cmd_desc):
        self.cmd_desc = new_cmd_desc

    def get_handler(self):
        return self.handler

    def set_handler(self, new_handler):
        self.handler = new_handler

    def get_handler_params(self):
        return self.handler_params

    def set_handler_params(self, new_handler_params):
        self.handler_params = new_handler_params

    def make_cli_cmd_string(self):
        # replace first occurrence of shorthand with shorthand in square brackets
        return self.cmd_name.replace(self.cmd_shorthand, f'[{self.cmd_shorthand}]', 1)

    @classmethod
    def get_commands(cls):
        return cls.commands.values()

    @classmethod
    def get_commands_dict(cls):
        return cls.commands

    @classmethod
    def get_command_names(cls):
        return cls.commands.keys()

    @classmethod
    def get_command_descriptions(cls, with_names=False):
        if with_names:
            return ((cmd.cmd_name, cmd.cmd_desc) for cmd in cls.commands.values())
        return map(lambda cmd: cmd.cmd_desc, cls.commands.values())

    @classmethod
    def get_command_shorthands(cls, with_names=False):
        if with_names:
            return ((cmd.cmd_name, cmd.cmd_shorthand) for cmd in cls.commands.values())
        return map(lambda cmd: cmd.cmd_shorthand, cls.commands.values())

    @classmethod
    def get_cmd_name_by_shorthand(cls, cmd_shorthand):
        for cur_name, cur_shorthand in cls.get_command_shorthands(with_names=True):
            if cur_shorthand == cmd_shorthand:
                break
        else:
            raise ValueError(f"no command with shorthand {cmd_shorthand} found")
        return cur_name

    @classmethod
    def remove_command(cls, cmd_name):
        # TODO: needed?
        try:
            cls.commands.pop(cmd_name)
        except KeyError:
            log_error(f"could not remove unknown command '{cmd_name}'")

    @classmethod
    def get_command(cls, cmd_name):
        try:
            cmd = cls.commands[cmd_name]
        except KeyError:
            log_error(f"could not remove unknown command '{cmd_name}'")
            return None
        return cmd


class Commands:
    process_images = Command('process images', 'process new images', 'p')
    edit_labels = Command('edit labels', 'edit labels of faces or clusters', 'e')
    reclassify = Command('reclassify', 'reclassify individuals', 'c')
    reset_cluster_ids = Command('reset cluster ids', 'reset the cluster ids', 'r')
    view_person = Command('view person', 'view person', 'v')
    clear_data = Command('clear data', 'clear (some of) the processed data', 'd')
    exit = Command('exit', 'exit', 'exit')

    @classmethod
    def initialize(cls):
        cls.process_images.set_handler(process_image_dir)
        cls.edit_labels.set_handler(edit_labels)
        cls.reclassify.set_handler(reclassify)
        cls.reset_cluster_ids.set_handler(reset_cluster_ids)
        cls.view_person.set_handler(view_person)
        cls.clear_data.set_handler(clear_data)

from functools import partial

from PIL import Image

from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.misc_helpers import get_user_decision, clean_string, wait_for_any_input, enumerate_strs


def get_directory_decision():
    pass


def view_person(clusters, **kwargs):
    """
    1. Fetch which labels exist (incl. Unknown Person)
    2. Prompt user, which person/label they would like to view
    3. Fetch all image names/paths for that person
    4. Prompt user, which image they would like to view
    5. Show image
    6. Go to 2.

    :param clusters:
    :param kwargs:
    :return:
    """
    # TODO: Make user choose file *name*, not path (and just inform them of the path they're on beforehand)
    # TODO: When only one choice (to pick path or image), make choice for user and inform them about it!
    # TODO: Refactor? (Extract functions)
    # TODO: Give option of renaming a file/directory?
    #       --> Best practices? How to do so *safely*?!)
    # TODO: How to include thumbnails and face ids in all of this?
    #       --> Give option to switch to/from edit_handler?

    get_label_decision = partial(get_user_decision, 'Would you like to select another person?')
    get_image_decision = partial(get_user_decision, 'Would you like to view another image of the person from this'
                                                    ' directory?')
    get_directory_decision = partial(get_user_decision, 'Would you like to select another directory containing images'
                                                        ' of the person?')
    cluster_labels = clusters.get_cluster_labels(unique=True)  # TODO: faster to use DB??
    # TODO: Extract some loop constructs as functions?
    # TODO: Are these interactions alright?
    # TODO: Catch errors!

    continue_label = ''
    while continue_label != 'n':
        chosen_label = user_choose_label(cluster_labels)
        if chosen_label is None:
            continue_label = get_label_decision()
            continue
        person_dir_paths_to_img_ids = DBManager.get_dir_paths_to_img_ids(chosen_label)
        person_dir_paths = person_dir_paths_to_img_ids.keys()

        continue_directory = ''
        while continue_directory != 'n':
            chosen_directory_path = user_choose_directory_path(person_dir_paths)
            if chosen_directory_path is None:
                continue_directory = get_directory_decision()
                continue

            image_ids = person_dir_paths_to_img_ids[chosen_directory_path]
            file_name_to_path_dict = DBManager.get_image_name_to_path_dict(chosen_directory_path, image_ids)

            continue_image = ''
            while continue_image != 'n':
                print(f"The currently chosen path is: '{chosen_directory_path}'.")
                chosen_image_path = user_choose_image_path(file_name_to_path_dict)
                if chosen_image_path is None:
                    continue_image = get_image_decision()
                    continue
                chosen_image = Image.open(chosen_image_path)
                chosen_image.show()
                continue_image = get_image_decision()
            continue_directory = get_directory_decision()
        continue_label = get_label_decision()


def user_choose_label(labels):
    return user_choose_func(labels, "a person's name", choice_with_nums=True)


def user_choose_image_path(file_name_to_path_dict):
    file_names = file_name_to_path_dict.keys()
    chosen_file_name = user_choose_func(file_names, "a file name", choice_with_nums=True)
    if chosen_file_name is None:
        return None
    chosen_image_path = file_name_to_path_dict[chosen_file_name]
    return chosen_image_path


def user_choose_directory_path(directory_paths):
    return user_choose_func(directory_paths, "a directory path", choice_with_nums=True)


def user_choose_func(valid_inputs, obj_name, choice_with_nums=False, print_obj_name=False, print_sorted_inputs=True):
    """

    :param choice_with_nums:
    :param valid_inputs: an iterable of strings
    :param obj_name:
    :param print_obj_name:
    :param print_sorted_inputs:
    :return:
    """
    # TODO: when to sort, when to handle nums to dict etc?!

    # TODO: Print error msg when sth doesn't work!
    # TODO: Refactor!!

    if print_sorted_inputs:
        valid_inputs = sorted(valid_inputs)
    if choice_with_nums:
        valid_inputs = dict(enumerate_strs(valid_inputs, start=1))
        obj_name = f"the number of {obj_name}"

    chosen_input = None
    while chosen_input not in valid_inputs:
        wait_for_any_input("\n"
                           f"Please enter {obj_name}, or press Enter to cancel. (Press Enter to continue)")
        if print_obj_name:
            print_valid_inputs(valid_inputs, obj_name, print_sorted=print_sorted_inputs)
        else:
            print_valid_inputs(valid_inputs, print_sorted=print_sorted_inputs)

        chosen_input = clean_string(input())
        if not chosen_input:
            chosen_input = None
            break

    if choice_with_nums and chosen_input is not None:
        chosen_input_value = valid_inputs[chosen_input]
        return chosen_input_value
    return chosen_input


def print_valid_inputs(valid_inputs, obj_name=None, choice_with_nums=False, print_sorted=True):
    # TODO: Refactor
    # TODO: Custom sorting with Unknown Person being first/last?
    obj_name_str = ' ' if obj_name is None else f' {obj_name} '

    if isinstance(valid_inputs, dict):
        valid_inputs_strs = (f'- [{input_key}]{obj_name_str}{valid_input}'
                             for input_key, valid_input in valid_inputs.items())
    else:
        valid_inputs_strs = (f'-{obj_name_str}{valid_input}'
                             for valid_input in valid_inputs)
    print('\n'.join(valid_inputs_strs))

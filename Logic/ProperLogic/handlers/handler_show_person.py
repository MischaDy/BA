from functools import partial

from PIL import Image

from Logic.ProperLogic.database_modules.database_logic import DBManager
from Logic.ProperLogic.misc_helpers import get_user_decision, clean_str, wait_for_any_input


def get_directory_decision():
    pass


def show_person(clusters, **kwargs):
    """
    1. Fetch which labels exist (incl. Unknown Person)
    2. Prompt user, which person/label they would like to be shown
    3. Fetch all image names/paths for that person
    4. Prompt user, which image they would like to view
    5. Show image
    6. Go to 2.

    :param clusters:
    :param kwargs:
    :return:
    """
    # TODO: Refactor? (Extract functions)
    # TODO: Give option of renaming a file/directory?
    #       --> Best practices? How to do so *safely*?!)
    # TODO: How to include thumbnails and face ids in all of this?
    #       --> Give option to switch to/from edit_handler?

    get_label_decision = partial(get_user_decision, 'Would you like to select another person?')
    get_image_decision = partial(get_user_decision, 'Would you like to view another image of the person?')
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
        person_directory_paths = DBManager.get_dir_paths_with_img_ids(chosen_label)

        continue_directory = ''
        while continue_directory != 'n':
            chosen_directory_path = user_choose_directory_path(person_directory_paths)
            if chosen_directory_path is None:
                continue_directory = get_directory_decision()
                continue
            person_image_paths = DBManager.get_image_paths(chosen_directory_path)

            continue_image = ''
            while continue_image != 'n':
                chosen_image_path = user_choose_image_path(person_image_paths)
                if chosen_image_path is None:
                    continue_image = get_image_decision()
                    continue
                chosen_image = Image.open(chosen_image_path)
                chosen_image.show()
                continue_image = get_image_decision()
            continue_directory = get_directory_decision()
        continue_label = get_label_decision()


def user_choose_label(labels):
    return user_choose_func(labels, "a person's name")


def user_choose_image_path(image_paths):
    return user_choose_func(image_paths, "an image path")


def user_choose_directory_path(directory_paths):
    # TODO: Rename object to sth more descriptive than that?
    return user_choose_func(directory_paths, "a directory path")


def user_choose_func(valid_inputs, obj_name):
    # TODO: Refactor!!
    chosen_input = None
    while chosen_input not in valid_inputs:
        wait_for_any_input(f"Please enter {obj_name}, or press Enter to cancel. (Press Enter to continue)")
        print_valid_inputs(valid_inputs, obj_name)
        chosen_input = clean_str(input())
        if not chosen_input:
            chosen_input = None
            break
    return chosen_input


def print_valid_inputs(valid_inputs, obj_name=' '):
    valid_inputs_strs = map(lambda valid_input: f'-{obj_name} {valid_input}', valid_inputs)
    print('\n'.join(valid_inputs_strs))

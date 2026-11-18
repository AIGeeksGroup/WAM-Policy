"""The ten QLM-Bench quadruped loco-manipulation tasks and their language instructions."""

TASKS = ("push_box", "press_button", "lift_basket", "kick_ball_to_goal",
         "sweep_block_to_zone", "close_fridge_door", "close_oven_door",
         "turn_valve", "pull_open_drawer", "open_room_door")

INSTRUCTIONS = {
    "push_box": "Push the box inside the outlined area.",
    "press_button": "Press the button.",
    "lift_basket": "Lift the basket by its handle.",
    "kick_ball_to_goal": "Kick the ball through the goal.",
    "sweep_block_to_zone": "Sweep the block into the outlined area.",
    "close_fridge_door": "Push against the fridge door until the door closes.",
    "close_oven_door": "Push the oven door until the door closes.",
    "turn_valve": "Rotate the valve lever upward to the upright position.",
    "pull_open_drawer": "Slide your foot into the drawer handle opening and pull the drawer outward.",
    "open_room_door": "Press down the lever handle, then push the door open.",
}

assert tuple(INSTRUCTIONS) == TASKS

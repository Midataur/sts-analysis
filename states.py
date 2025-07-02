from game_data import *

def handle_upgrade(card):
    # get existing upgrade
    count = 0
    base_card = card

    if "+" in card:
        base_card, count = card.split("+")
        count = int(count)
    
    return base_card, count

def format_fight(fight):
    obtained = [] if fight["picked"] == "SKIP" else [fight["picked"]]

    return {
        "cards_obtained": obtained,
        "cards_removed": [],
        "floor": fight["floor"],
        "meta": fight
    }

def format_event(event):
    removed = []
    obtained = []

    if "cards_removed" in event.keys():
        removed += event["cards_removed"]
    
    if "cards_transformed" in event.keys():
        removed += event["cards_transformed"]

    if "cards_obtained" in event.keys():
        obtained += event["cards_obtained"]

    if "cards_upgraded" in event.keys():
        for card in event["cards_upgraded"]:
            base_card, count = handle_upgrade(card)

            removed.append(card)
            obtained.append(base_card+f"+{count+1}")

    return {
        "cards_removed": removed,
        "cards_obtained": obtained,
        "floor": event["floor"],
        "meta": event
    }

def format_campfire(campfire):
    removed = []
    obtained = []

    if campfire["key"] == "SMITH":
        base_card, count = handle_upgrade(campfire["data"])

        removed.append(campfire["data"])
        obtained.append(base_card+f"+{count+1}")
    
    if campfire["key"] == "PURGE":
        removed.append(campfire["data"])
    
    return {
        "cards_removed": removed,
        "cards_obtained": obtained,
        "floor": campfire["floor"],
        "meta": campfire
    }

def get_relics(run, floor):
    relics = []

    for gained in run["relics_obtained"]:
        if gained["floor"] <= floor:
            relics.append(gained["key"])
    
    return relics

def construct_timeline(run):
    # handle fight
    fights = list(map(format_fight, run["card_choices"]))

    # handle events
    events = list(map(format_event, run["event_choices"]))

    # handle campfire
    fires = list(map(format_campfire, run["campfire_choices"]))

    # handle shop
    shop_gains = [
        {
            "floor": x, "cards_obtained": [y], "cards_removed": []
        } for x,y in zip(run["item_purchase_floors"], run["items_purchased"])
    ]

    shop_losses = [
        {
            "floor": x, "cards_removed": [y], "cards_obtained": []
        } for x,y in zip(run["items_purged_floors"], run["items_purged"])
    ]

    timeline = (
        fights +
        events +
        fires +
        shop_gains +
        shop_losses
    )

    # filter and sort by floors
    timeline.sort(key=lambda x: -x["floor"])
    return timeline

# deck is reconstructed backward from the final state
# final result isn't perfect, but is pretty close
# there's too many edgecases to get perfect results
def reconstruct_state(run, floor, verbose=False):
    #make sure floor is the right data type
    floor = int(floor)

    character = run["character_chosen"]

    # construct timeline
    timeline = construct_timeline(run)

    # filter timeline
    timeline = list(filter(lambda x: x["floor"] > floor, timeline))

    # reconstruct relics
    relics = get_relics(run, floor)

    # get the deck at that floor
    deck = list(run["master_deck"])

    for event in timeline:
        # print(deck)
        # print(event, "\n")
        if verbose:
            print(f"Floor {event["floor"]}:")
            print("Gained:", ", ".join(event["cards_obtained"]))
            print("Lost:", ", ".join(event["cards_removed"]), "\n")

        # since we reconstruct backwards, swap obtains and removes
        if "cards_removed" in event.keys():
            for card in event["cards_removed"]:
                if card not in RELICS_LIST:
                    deck.append(card)

        if "cards_obtained" in event.keys():
            for card in event["cards_obtained"]:
                if card not in RELICS_LIST:
                    # deal with unrecorded upgrades
                    base_card, count = handle_upgrade(card)
                    upgraded = base_card+f"+{count}"

                    if card in deck:
                        deck.remove(card)
                    elif upgraded in deck:
                        deck.remove(upgraded)

    capped_floor = min(floor, run["floor_reached"])

    # deal with some edge cases:
    if (
        "Necronomicon" in run["relics"] and 
        "Necronomicon" not in relics and
        "Necronomicurse" in deck
    ):
        deck.remove("Necronomicurse")

    return {
        "character": character,
        "current_hp": run["current_hp_per_floor"][capped_floor],
        "max_hp": run["max_hp_per_floor"][capped_floor],
        "gold": run["gold_per_floor"][capped_floor],
        "deck": deck,
        "relics": relics,
        "ascension": run["ascension_level"],
        "floor": floor
    }
import json

import nluconfig as config


def nice_json(entities):
    for ent in entities:
        # remove span
        ent.pop('span')
        # add entity description to type_id
        ent_type_id = int(ent['type_id'])
        index       = ent_type_id - 1
        if 0 <= index and index < len(config.ENTITY_MEANINGS) :
            ent_type_name = config.ENTITY_MEANINGS[index]
            ent['type_id'] = f"{ent_type_id} ({ent_type_name})"
    return entities

def print_output(text, intent, entities):
    print(f"Input   : {text}")
    print(f"Intent  : {intent} ({config.INTENT_MEANINGS[intent]})")

    entities = nice_json(entities)
    print(f"Entities:", json.dumps(entities, indent=2, ensure_ascii=False))
    print(f"")
    print(f"input > ", end='', flush=True)

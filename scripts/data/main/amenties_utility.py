import pandas as pd
from collections import Counter

def create_amenity_features(amenities_col):
    df = pd.DataFrame()

    # Clean and split amenities.
    def clean_split(x):
        if pd.isna(x):
            return set()
        return set(a.strip().lower() for a in x.split(','))
    
    amenities_sets = amenities_col.apply(clean_split)

    # Weighted luxury score.
    luxury_weights = {
        'pool': 3,
        'hot tub': 3,
        'gym': 2,
        'beach access': 3,
        'waterfront': 3,
        'elevator': 2,
        'bbq grill': 2,
        'outdoor kitchen': 3,
        'sauna': 3,
        'resort access': 3,
        'hot tub': 3,
    }

    df['luxury_score'] = amenities_sets.apply(
        lambda x: sum(weight for amenity, weight in luxury_weights.items() if amenity in x)
    )

    # Category groups.
    basic = ['wifi', 'kitchen', 'refrigerator', 'essentials']
    
    comfort = ['air conditioning', 'heating', 'tv', 'portable fans']
    
    kitchen = ['microwave', 'stove', 'oven', 'coffee maker', 
               'cooking basics', 'dishes and silverware']
    
    safety = ['smoke alarm', 'fire extinguisher', 
              'first aid kit', 'carbon monoxide alarm']
    
    outdoor = ['patio or balcony', 'backyard', 'outdoor furniture', 
               'bbq grill', 'hammock']
    
    location = ['beach access', 'waterfront', 'lake access']
    
    family = ['crib', 'high chair', "children's dinnerware", 
              'baby bath', 'board games']
    
    services = ['long term stays allowed', 'luggage dropoff allowed', 
                'cleaning before checkout']
    
    categories = {
        'basic': basic,
        'comfort': comfort,
        'kitchen': kitchen,
        'safety': safety,
        'outdoor': outdoor,
        'location': location,
        'family': family,
        'services': services
    }

    # Skip 'location' category to avoid multicollinearity with has_beach_access
    for cat_name, items in categories.items():
        if cat_name == 'location':
            continue
        df[f'{cat_name}_count'] = amenities_sets.apply(
            lambda x: sum(1 for item in items if item in x)
        )

    # Key binary flags.
    key_flags = [
        'pool', 'hot tub', 'gym', 'beach access',
        'dedicated workspace', 'pets allowed',
        'free parking on premises', 'air conditioning'
    ]

    for flag in key_flags:
        df[f'has_{flag.replace(" ", "_")}'] = amenities_sets.apply(
            lambda x: int(flag in x)
        )

    # Total amenities.
    df['total_amenities'] = amenities_sets.apply(len)

    # Ratio features.
    df['comfort_to_total_ratio'] = (df['comfort_count'] / (df['total_amenities'] + 1)).round(2)


    # Rarity score.
    all_amenities = [a for s in amenities_sets for a in s]
    freq = Counter(all_amenities)

    df['rarity_score'] = amenities_sets.apply(
        lambda x: sum(1 / freq[a] for a in x if a in freq)
    )

    return df
import pandas as pd

def categorize_listing(x):
    if pd.isna(x):
        return 'unknown'
    
    x = str(x).lower().strip()

    # Hospitality listing types.
    if any(k in x for k in [
        'hotel', 'hostel', 'resort', 'bed and breakfast', 'bnb', 'pension',
        'riad', 'casa particular', 'ryokan', 'inn', 'lodge', 'guesthouse',
        'guest house', 'guest suite', 'boutique hotel', 'aparthotel'
    ]):
        return 'hospitality'

    # Nature stays.
    elif any(k in x for k in [
        'farm', 'cabin', 'cottage', 'chalet', 'hut', 'barn', 'ranch',
        'nature lodge', 'eco lodge', 'farm stay', 'tiny home', 'earth home'
    ]):
        return 'nature_stay'

    # Unique stays.
    elif any(k in x for k in [
        'treehouse', 'boat', 'houseboat', 'camper', 'rv', 'tent', 'yurt',
        'dome', 'campsite', 'shipping container', 'container', 'island',
        'bus', 'train', 'plane', 'cave'
    ]):
        return 'unique_stay'

    # Standard property types.
    elif any(k in x for k in ['apartment', 'rental unit', 'flat']):
        return 'apartment'

    elif any(k in x for k in ['home', 'house', 'residential']):
        return 'home'

    elif any(k in x for k in ['condo', 'condominium']):
        return 'condo'

    elif 'villa' in x:
        return 'villa'

    elif 'bungalow' in x:
        return 'bungalow'

    elif 'loft' in x:
        return 'loft'

    elif 'townhouse' in x:
        return 'townhouse'

    # Luxury and rare types.
    elif any(k in x for k in [
        'castle', 'palace', 'mansion',
        'penthouse', 'resort', 'boutique',
        'luxury', 'premium', 'exclusive',
        'upscale', 'high-end'
    ]):
        return 'luxury_unique'

    # Unknown or noise labels.
    elif x in ['unknown', '', 'other']:
        return 'unknown'

    # Fallback.
    else:
        return 'other'
    
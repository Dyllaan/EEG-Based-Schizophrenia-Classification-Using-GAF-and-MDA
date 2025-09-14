import os
import pickle
import numpy as np

def save_cache(data, cache_file, data_type="data"):
    """Universal caching function."""
    if isinstance(data, tuple) and len(data) == 2:
        cache_data = {
            'data': data[0],
            'labels': data[1],
            'num_samples': len(data[0]),
            'data_shape': data[0][0].shape if len(data[0]) > 0 else None,
            'type': data_type
        }
    else:
        cache_data = {'data': data, 'type': data_type}
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Cached {len(cache_data.get('data', []))} {data_type} to {cache_file}")

def load_cache(cache_file, data_type="data"):
    """Universal cache loading function."""
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        data = cache_data['data']
        labels = cache_data.get('labels')
        
        print(f"Loaded {len(data)} cached {data_type} from {cache_file}")
        if labels is not None:
            print(f"Label distribution: {np.bincount(labels)}")
            return data, labels
        return data
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None
    
def clear_cache():
    """Remove cache files to force reload from CSV files."""
    cache_files = ['processed_eeg_cache.pkl', 'gaf_images_cache.pkl']
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed {cache_file}")
    print("Cache cleared. Next run will reload from CSV files.")

def save_processed_data(time_series_data, labels, cache_file='processed_eeg_cache.pkl'):
    save_cache((time_series_data, labels), cache_file, data_type="processed_eeg")

def load_processed_data(cache_file='processed_eeg_cache.pkl'):
    result = load_cache(cache_file, data_type="processed_eeg")
    if result is None:
        return None, None
    time_series_data, labels = result
    return time_series_data, labels
    
def save_gaf_images(images, labels, cache_file='gaf_images_cache.pkl'):
    save_cache((images, labels), cache_file, data_type="gaf_images")

def load_gaf_images(cache_file='gaf_images_cache.pkl'):
    result = load_cache(cache_file, data_type="gaf_images")
    if result is None:
        return None, None
    images, labels = result
    print(f"Image shape: {images[0].shape if len(images) > 0 else None}")
    return images, labels
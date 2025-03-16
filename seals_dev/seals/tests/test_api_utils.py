import hazelbean as hb
import os
from seals import seals_api_parsing

def test_api_utils():
    print('test_api_utils')
    
    input_path = '../seals_api_input.md'
    api_dict = seals_api_parsing.parse_input_api_md(input_path)
    
    df = seals_api_parsing.api_dict_to_df(api_dict)
    
    # seals_api_parsing.iterate_scenarios(df)
    
if __name__ == '__main__':
    test_api_utils()
    
    print('Tests complete')
    done = 1
    
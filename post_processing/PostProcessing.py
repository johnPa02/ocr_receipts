from rotation_corrector.utils.utility import cer_loss_one_image
from rotation_corrector.utils.spell_check import fix_datetime
from config import dictionary_path
import re


def swap_total_cost(total_cost):
    # Use regular expressions to separate the number and the text
    match = re.match(r'([\d.,]+)\s*-\s*(.+)', total_cost)
    if match:
        number = match.group(1)
        text = match.group(2)
        # Swap the positions
        return f"{text.strip()} - {number.strip()}"
    return total_cost  # Return as-is if the format doesn't match

def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        dictionary = eval(file.read())
    return dictionary


class PostProcessor:
    def __init__(self):
        """
        Initialize the PostProcessing with an optional dictionary for corrections.

        :param dictionary: A dictionary containing 'seller' and 'address' data for corrections.
        """
        self.dictionary = load_dictionary(dictionary_path)

    def process(self, kie_result):
        """
        Process the KIE result dictionary.

        :param kie_result: A dictionary with keys 'ADDRESS', 'SELLER', 'TIMESTAMP', and 'TOTAL_COST'.
        :return: Processed dictionary with potentially corrected values.
        """
        if self.dictionary:
            self._fix_result_by_dictionary(kie_result)

        self._fix_result_by_rule_based(kie_result)
        return kie_result

    def _fix_result_by_dictionary(self, result_dict):
        """
        Fixes the SELLER and ADDRESS fields in the result_dict using the provided dictionary.

        :param result_dict: A dictionary with keys 'ADDRESS', 'SELLER', 'TIMESTAMP'
        """
        list_seller = self.dictionary.get('seller', [])
        list_address = self.dictionary.get('address', [])

        # SELLER processing
        seller_str = result_dict.get('SELLER', '')
        min_cer = 1
        min_seller = None
        for seller in list_seller:
            seller_ori_str = ' '.join(seller['SELLER'])
            cer = cer_loss_one_image(seller_str, seller_ori_str)
            if cer < min_cer:
                min_cer = cer
                min_seller = seller['SELLER']
        if min_cer < 0.3 and min_cer > 0:
            result_dict['SELLER'] = ' '.join(min_seller)

        # ADDRESS processing
        address_str = result_dict.get('ADDRESS', '')
        min_cer = 1
        min_address = None
        for address in list_address:
            address_ori_str = ' '.join(address['ADDRESS'])
            cer = cer_loss_one_image(address_str, address_ori_str)
            if cer < min_cer:
                min_cer = cer
                min_address = address['ADDRESS']
        if min_cer < 0.3 and min_cer > 0:
            result_dict['ADDRESS'] = ' '.join(min_address)

    def _fix_result_by_rule_based(self, result_dict):
        """
        Applies rule-based fixes to the TIMESTAMP and TOTAL_COST fields.

        :param result_dict: A dictionary with keys 'ADDRESS', 'SELLER', 'TIMESTAMP', and 'TOTAL_COST'.
        """
        # TIMESTAMP processing
        if 'TIMESTAMP' in result_dict:
            result_dict['TIMESTAMP'] = fix_datetime(result_dict['TIMESTAMP'])
        # TOTAL_COST processing (not implemented yet)
        result_dict['TOTAL_COST'] = swap_total_cost(result_dict['TOTAL_COST'])

TOTAL_COST_chars = '0123456789-,.đdvn '
TOTAL_COST_keys = ['TỔNG TIỀN PHẢI T.TOÁN', 'Cộng tiền hàng', 'Tổng cộng', 'Tong cong', 'Tổng cộng (đã gồm VAT)',
                   'Tổng tiền', 'Tổng tiền hàng', 'TỔNG CỘNG', 'Tổng số tiền thanh toán', 'Tiền Thanh Toán',
                   'Thành Tiền', 'Total Amount', 'Total', 'Tổng thanh toán', 'Tại quầy', 'Tổng số thanh toán', 'Tổng hóa đơn (VNĐ)']

TIMESTAMP_keys = ['Ngày', 'Thời gian']
TIMESTAMP_noise_keys = ['Số HĐ', 'Số GD']

from utils.utility import cer_loss_one_image


def validate_TOTAL_COST_amount(input_str, thres=0.8):
    if len(input_str) == 0:
        return False
    count = 0
    input_str = input_str.lower()
    for ch in input_str:
        if ch in TOTAL_COST_chars:
            count += 1
    score = count / len(input_str)
    #print(score)
    return True if score > thres else False


def validate_TOTAL_COST_keys(input_str, cer_thres=0.2):
    lower_totalcost = input_str.lower()
    min_cer = 1
    for k in TOTAL_COST_keys:
        lower_k = k.lower()
        cer = cer_loss_one_image(lower_totalcost, lower_k)
        if cer < min_cer:
            min_cer = cer
    return True if min_cer < cer_thres else False


def validate_TIMESTAMP(input_str):
    lower_totalcost = input_str.lower()
    for k in TIMESTAMP_keys:
        lower_k = k.lower() +':'
        if lower_k in lower_totalcost:
            print(k, input_str)
            return True
    return False

def validate_SELLER(list_seller, input_str, cer_thres=0.2):
    if len(input_str) < 10:
        return False
    input_str = input_str.lower()
    min_cer = 1
    min_str = ''
    for s in list_seller:
        if s['count'] > 1:
            for line in s['SELLER']:
                lower_line = line.lower()
                cer = cer_loss_one_image(lower_line, input_str)
                if cer < min_cer:
                    min_cer = cer
                    min_str = lower_line
    if min_cer < cer_thres:
        print(round(min_cer, 2), input_str, '-----', min_str)
    return True if min_cer < cer_thres else False


def validate_ADDRESS(list_address, input_str, cer_thres=0.2):
    if len(input_str) < 10:
        return False
    input_str = input_str.lower()
    min_cer = 1
    min_str = ''
    for s in list_address:
        if s['count'] > 1:
            for line in s['ADDRESS']:
                lower_line = line.lower()
                cer = cer_loss_one_image(lower_line, input_str)
                if cer < min_cer:
                    min_cer = cer
                    min_str = lower_line
    if min_cer < cer_thres:
        print(round(min_cer, 2), input_str, '-----', min_str)
    return True if min_cer < cer_thres else False


def fix_datetime(input_str):  # for string len >42
    input_str = input_str.lstrip(' ').rstrip(' ')
    lower_input_str = input_str.lower()
    final_time_pos = -1
    for k in TIMESTAMP_keys:
        lower_k = k.lower()
        time_pos = lower_input_str.find(lower_k)
        if time_pos != -1:
            final_time_pos = time_pos

    final_noise_pos = -1
    for k in TIMESTAMP_noise_keys:
        lower_k = k.lower()
        noise_pos = lower_input_str.find(lower_k)
        if noise_pos != -1:
            final_noise_pos = noise_pos

    final_str = input_str
    if final_noise_pos > 0:
        final_str = input_str[:final_noise_pos]
    if final_time_pos > 0:
        final_str = input_str[final_time_pos:]

    final_str = final_str.lstrip(' ').rstrip(' ')

    return final_str

    # final_str = result_dict['TIMESTAMP'][0][time_pos:]
    # if ' -' not in final_str:
    #     final_str = final_str.replace('-', ' -')
    # if '- ' not in final_str:
    #     final_str = final_str.replace('-', '- ')
    # result_dict['TIMESTAMP'][0] = final_str


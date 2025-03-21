import json
from tqdm import tqdm
from datetime import datetime
import pickle
from dateutil.parser import parse

def str_to_datetime(date_str):
    try:
        return parse(date_str)
    except ValueError:
        # 处理无法解析的情况
        return None

def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict

def ana_user_PIaBI(filename):
    print(filename)
    # date_format = "%Y-%m-%d %H:%M:%S"

    u_count=0
    foutPIaBI = open(filename+ '_PIaBI_dict.pkl', 'wb')
    u_PIaBI_dic = {}

    tmpdict = parse_json_all(filename)

    u_count=0
    u_mar_count=0

    u4sc_dict={}
    u4sc_dict[0]={
        'count':0,
        'ori_hf': 0,
        'incI_hf':0,
        'laN_hf':0,
        'gen_m':0,
        'gen_f':0,
        'mar':0,
    }
    u4sc_dict[1]={
        'count':0,
        'ori_hf': 0,
        'incI_hf':0,
        'laN_hf':0,
        'gen_m':0,
        'gen_f':0,
        'mar':0,
    }
    u4sc_dict[2]={
        'count':0,
        'ori_hf': 0,
        'incI_hf':0,
        'laN_hf':0,
        'gen_m':0,
        'gen_f':0,
        'mar':0,
    }
    u4sc_dict[3]={
        'count':len(tmpdict),
        'ori_hf': 0,
        'incI_hf':0,
        'laN_hf':0,
        'gen_m':0,
        'gen_f':0,
        'mar':0,
    }

    for iuser in tqdm(tmpdict):
        # u4sc_dict[iuser['risk_label']]['count'] += 1
        u4sc_dict[int(iuser['dep_label'])]['count'] += 1

        PI_list=[]
        BI_list=[]

        if iuser["gender"] == "男":
            gen_sten="该用户的性别是男性。"
            # u4sc_dict[iuser['risk_label']]['gen_m']+=1
            u4sc_dict[int(iuser['dep_label'])]['gen_m']+=1

            u4sc_dict[3]['gen_m']+=1

        else:
            gen_sten="该用户的性别是女性。"
            # u4sc_dict[iuser['risk_label']]['gen_f']+=1
            u4sc_dict[int(iuser['dep_label'])]['gen_f']+=1

            u4sc_dict[3]['gen_f']+=1


        PI_list.append(gen_sten)

        twi_count=0
        ori_twi_count=0
        incIm_twi_count=0
        laN_twi_count=0
        marryflag=0

        mar_list=['我老公','我老婆','我丈夫','我妻子','我爱妻','我夫人','我结婚',
                  '我的老公','我的老婆','我的丈夫','我的妻子','我的爱妻','我的夫人',
                  ]

        for itwi in iuser["tweets"]:
            if itwi['tweet_is_original']=='True':
                ori_twi_count+=1
            if any(word in itwi['tweet_content'] for word in mar_list) and marryflag==0:
                marryflag = 1

            if itwi['posted_picture_url'] !='无' and itwi['posted_picture_url'] !=[]:
                incIm_twi_count+=1

            # dto = datetime.strptime(itwi['posting_time'], date_format)
            dto = str_to_datetime(itwi['post_time'])

            if 23<= dto.hour <=24 or 0<= dto.hour <=6:
                laN_twi_count+=1

            twi_count+=1

        if marryflag==0:
            mar_sten='该用户的婚姻状况是未知的'
        else:
            mar_sten='该用户的婚姻状况是已婚。'
            # u4sc_dict[iuser['risk_label']]['mar'] += 1
            u4sc_dict[int(iuser['dep_label'])]['mar'] += 1

            u4sc_dict[3]['mar'] += 1

        # PI_list.append(mar_sten)

        ori_rate=float(ori_twi_count/twi_count)
        incIm_rate=float(incIm_twi_count/twi_count)
        laN_rate=float(laN_twi_count/twi_count)

        if ori_rate>=0.5:
            # u4sc_dict[iuser['risk_label']]['ori_hf'] += 1
            u4sc_dict[int(iuser['dep_label'])]['ori_hf'] += 1

            u4sc_dict[3]['ori_hf'] += 1
        if incIm_rate>=0.5:
            # u4sc_dict[iuser['risk_label']]['incI_hf'] += 1
            u4sc_dict[int(iuser['dep_label'])]['incI_hf'] += 1

            u4sc_dict[3]['incI_hf'] += 1
        if laN_rate>=0.5:
            # u4sc_dict[iuser['risk_label']]['laN_hf'] += 1
            u4sc_dict[int(iuser['dep_label'])]['laN_hf'] += 1


            u4sc_dict[3]['laN_hf'] += 1

        ori_sten='该用户的原创帖子数量占总发帖数量的'+ str(round(ori_rate*100,2)) +'%。'
        incIm_sten='该用户的包含图像的帖子数量占总发帖数量的'+ str(round(incIm_rate*100,2)) +'%。'
        laN_sten='该用户在深夜发帖的次数占总发帖次数的'+ str(round(laN_rate*100,2)) +'%。'

        BI_list.append(ori_sten)
        BI_list.append(incIm_sten)
        BI_list.append(laN_sten)

        u_PIaBI_dic[u_count]={'PI':PI_list,'BI':BI_list}
        u_count+=1

    pickle.dump(u_PIaBI_dic, foutPIaBI)
    foutPIaBI.close()

    for itmp in [0,1,3]:
        print(itmp)
        print(u4sc_dict[itmp])
        all_c=u4sc_dict[itmp]['count']

        f_sten='女性占比'+ str(round(u4sc_dict[itmp]['gen_f']/all_c*100,2)) +'%。'
        print(f_sten)
        f_sten='男性占比'+ str(round(u4sc_dict[itmp]['gen_m']/all_c*100,2)) +'%。'
        print(f_sten)

        f_sten='原创过半占比'+ str(round(u4sc_dict[itmp]['ori_hf']/all_c*100,2)) +'%。'
        print(f_sten)
        f_sten='包含图片过半占比'+ str(round(u4sc_dict[itmp]['incI_hf']/all_c*100,2)) +'%。'
        print(f_sten)
        f_sten='深夜过半占比'+ str(round(u4sc_dict[itmp]['laN_hf']/all_c*100,2)) +'%。'
        print(f_sten)

        f_sten='结婚占比'+ str(round(u4sc_dict[itmp]['mar']/all_c*100,2)) +'%。'
        print(f_sten)

        print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
    dictswdd={'SWDD_1k8': {
        'train': './datasets/SWDD/SWDD_1k8_train.json',
        'test': './datasets/SWDD/SWDD_1k8_test.json'
    },
        'SWDD_3k7': {
            'train': './datasets/SWDD/SWDD_3k7_train.json',
            'test': './datasets/SWDD/SWDD_3k7_test.json'
        }, }
    ana_user_PIaBI(dictswdd['SWDD_1k8']['train'])
    ana_user_PIaBI(dictswdd['SWDD_1k8']['test'])

    ana_user_PIaBI(dictswdd['SWDD_3k7']['train'])
    ana_user_PIaBI(dictswdd['SWDD_3k7']['test'])

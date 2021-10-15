import pandas as pd
import numpy as np

RAW_CHALLENGE_PATH = "./data/nuMoM2b_Dataset_NICHD Data Challenge.csv"
PRESELECTED_FEATURE_PATH  = "./preprocessing/preeclampsia_features.csv"
OUTPUT_PATH = "./data/"

def get_raw_data():
    data = pd.read_csv(RAW_CHALLENGE_PATH, na_values=["M","D","R","S","E","N"])
    return data

def get_dataset(data, visit):
    feat_list = pd.read_csv(PRESELECTED_FEATURE_PATH)

    # working with only the patient wilt he analytes available
    data = data[~data.PAPPA_1.isnull() | ~data.PAPPA.isnull()]

    if visit ==  4:
        pre_feat = list(feat_list["Feature in numom2b"].values)
    elif visit == 3:
        pre_feat = list(feat_list.loc[feat_list["visit 4"] == 0, "Feature in numom2b"].values)
    elif visit == 2:
        pre_feat = list(feat_list.loc[(feat_list["visit 4"] == 0) & (feat_list["visit 3"] == 0), "Feature in numom2b"].values)
    elif visit == 1:
        pre_feat = list(feat_list.loc[(feat_list["visit 4"] == 0) & (feat_list["visit 3"] == 0) & (feat_list["visit 2"] == 0), "Feature in numom2b"].values)


    pre_feat.append("PublicID")
    pre_feat.append("PEgHTN")

    new_data = data.loc[:, data.columns.isin(pre_feat)]

    # remove all gestational hypertension and superimposed patient from the datset
    new_data = new_data[~new_data.PEgHTN.isin([5,6, 2, 4])]
    new_data = new_data[~new_data.PEgHTN.isnull()]

    # generate outcomes
    new_data["OUTCOME"] = 1
    new_data.loc[new_data.PEgHTN == 7, "OUTCOME"] = 0
    new_data = new_data.drop("PEgHTN", axis=1)


    # calculate blood presure information  
    if visit > 2:
        new_data["MAP_V3"] = (new_data["V3BA02a1"] + 2 * new_data["V3BA02b1"])/3
        new_data["Diastolic_CHG_V3_2"] = new_data["V3BA02b1"]/new_data["V1BA06b1"] - 1
        new_data["Systolic_CHG_V3_2"] = new_data["V3BA02a1"]/new_data["V1BA06a1"] - 1
        new_data["Systolic_CHG_V3"] = new_data["V3BA02a1"]/new_data["V2BA02a1"] - 1
        new_data["Diastolic_CHG_V3"] = new_data["V3BA02b1"]/new_data["V2BA02b1"] - 1
    elif visit  > 1:
        new_data["MAP_V2"] = (new_data["V2BA02a1"] + 2 * new_data["V2BA02b1"])/3
        new_data["Systolic_CHG_V2"] = new_data["V2BA02a1"]/new_data["V1BA06a1"] - 1
        new_data["Diastolic_CHG_V2"] = new_data["V2BA02b1"]/new_data["V1BA06b1"] - 1
    elif visit == 1:
        for analyte in ['ADAM12', 'ENDOGLIN', 'VEGF', 'AFP', 'fbHCG', 'INHIBINA']:
            new_data.loc[new_data.PublicID.isin(data[data.Visit_Number == 2].PublicID), analyte] = np.nan


    new_data["MAP_V1"] = (new_data["V1BA06a1"] + 2 * new_data["V1BA06b1"])/3
    print(new_data.shape)
    new_data.to_csv(OUTPUT_PATH + "V{}.csv".format(visit), index=False)


if __name__ == "__main__":
    data = get_raw_data()
    for x in range(1, 5):
        get_dataset(data, x)



import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import utils

class Recording:
    conditions = ["gaze", "headAndGaze", "nod", "smoothPursuit"]
    parameter = {
        "gaze": ["DwellTime"],
        "headAndGaze": ["HeadGazeCorrection", "DwellTime"],
        "nod": ["DegreeEnd", "DegreeMove"],
        "smoothPursuit": ["CorrelationTH", "Shape"]
    }

    group_definition = {
        1: ["gaze", "headAndGaze", "nod", "smoothPursuit"],
        2: ["gaze", "nod", "headAndGaze", "smoothPursuit"],
        3: ["gaze", "nod", "smoothPursuit", "headAndGaze"],
        4: ["gaze", "smoothPursuit", "nod", "headAndGaze"],
        5: ["gaze", "smoothPursuit", "headAndGaze", "nod"],
        6: ["gaze", "headAndGaze", "smoothPursuit", "nod"]
    }
    
    
    @staticmethod
    def _read_user_informations(path):
        dat = pd.read_table(os.path.join(path, [x for x in os.listdir(path) if ".txt" in x][0]))
        name, age, gender, etk_exp, vr_exp = dat.loc[0, ["Name", "Age", "Gender", "ET-Experience?", "VR-Experience?"]]
        return name, age, gender, etk_exp, vr_exp

    @staticmethod
    def _read_condition(path, condition):
        files = os.listdir(os.path.join(path, condition))
        scene_data = pd.read_table(os.path.join(path, condition, "SceneData.tsv"), sep="\t", decimal=",")
        gaze_file = "gaze_event.csv" if "gaze_event.csv" in files else "gaze.csv"
        gaze = pd.read_table(os.path.join(path, condition, gaze_file), sep="\t", decimal=",")
        gaze = gaze.loc[gaze.isValid]
        gaze120_file = "gaze_120fps_event.csv" if "gaze_120fps_event.csv" in files else "gaze_120fps.csv"
        gaze120 = pd.read_table(os.path.join(path, condition, gaze120_file), sep="\t", decimal=",")
        gaze120 = gaze120.loc[gaze120.isValid]
        round_data = pd.read_table(os.path.join(path, condition, "RoundData.tsv"), sep="\t", decimal=",")
        # files = os.listdir(os.path.join(path, condition))
        # questionnaire = pd.read_table(os.path.join(path, condition, [x for x in files if "TLX" in x][0]), sep=";")
        return {
            "SceneData": scene_data,
            "Gaze": gaze,
            "Gaze120": gaze120,
            "RoundData": round_data,
            # "Questionnaire": questionnaire
        }
    
    @staticmethod
    def _read_questionnaires(path):
        answers = dict()
        for condition in Recording.conditions:
            file = [x for x in os.listdir(os.path.join(path, condition)) if "TLX" in x][0]
            temp = pd.read_table(os.path.join(path, condition, file), sep=";")
            answers[condition] = dict()
            for _, row in temp.iterrows():
                answers[condition][row.QuestionName] = row.Value
        return pd.DataFrame(answers)
    
    def __init__(self, path) -> None:
        self.path = path
        self.name, self.age, self.gender, self.et_exp, self.vr_exp = Recording._read_user_informations(path)
        self.pref_method = None
        self.visual_aid = None
        self.data = {c: Recording._read_condition(path, c) for c in Recording.conditions}
        self.answers = Recording._read_questionnaires(path)
        self.summarized_final_rounds = {cond: None for cond in Recording.conditions}
        self.group = self._check_group()

    def __getitem__(self, index):
        if index in Recording.conditions:
            return self.data[index]
        if index == "answers":
            return self.answers
        return None
    
    def __repr__(self):
        return f"Recording of {self.name}; Age {self.age}; Gender {self.gender}; ET experience {self.et_exp}; VR experience {self.vr_exp}; Visual Aid {self.visual_aid}; Preferred method {self.pref_method}."
    
    def _check_group(self):
        timestamps = list()
        method = list()
        for cond in Recording.conditions:
            method.append(cond)
            timestamps.append(self[cond]["Gaze"].iloc[0]["System Timestamp"])
        method = [method[i] for i in np.argsort(timestamps)]
        for k in Recording.group_definition:
            if method == Recording.group_definition[k]:
                return k
        print(f"[WARNING] Participant {self.name} has no group!")
        return -1

    def get_rounds(self, condition, exclude_early_stops=True):
        round_data = self[condition]["RoundData"]
        if exclude_early_stops:
            return len(round_data.loc[~round_data.EarlyStop])
        return len(round_data)
    
    def get_round_gaze(self, round, condition, fps120=False):
        gaze = self[condition]["Gaze"] if not fps120 else self[condition]["Gaze120"]
        roundData = self[condition]["RoundData"]
        start, end = roundData.loc[roundData.Round==round, ["StartDeviceTime", "EndDeviceTime"]]
        return gaze.loc[(gaze["Device Timestamp"] > start) & (gaze["Device Timestamp"] < end)]

    def summarize_rounds(self, condition, force_recalculation=False):
        if self.summarized_final_rounds[condition] is not None and not force_recalculation:
            return self.summarized_final_rounds[condition]
        round_data = self.get_final_rounds(condition)
        scene_data = self[condition]["SceneData"]
        etk_stats = "Event" in self[condition]["Gaze120"].columns
        if etk_stats:
            gaze_data = self[condition]["Gaze120"]
        results = dict()
        for _, row in round_data.iterrows():
            t = None
            dat = scene_data.loc[(scene_data.Round==row["Round"]) & ~scene_data.Msg.isna(), ["Timestamp", "Msg"]]
            results[row["Round"]] = {"Correct": row.CorrectDestroyed, "Incorrect": row.IncorrectDestroyed, "Points": row.Points}
            for para in Recording.parameter[condition]:
                results[row["Round"]][para] = row[para]
            time_on_task = list()
            for _, rowDat in dat.iterrows():
                if rowDat.Msg == "StartGame":
                    t = rowDat.Timestamp
                if "MainTargetDestroyed" in rowDat.Msg:
                    if t is not None:
                        time_on_task.append(rowDat.Timestamp - t - 1) # -1 because each robot needs one second do arrive the destination
                        t = rowDat.Timestamp
            results[row["Round"]]["MeanTimeOnTask"] = np.mean(time_on_task)
            results[row["Round"]]["StdTimeOnTask"] = np.std(time_on_task)
            results[row["Round"]]["TimeOnTask"] = time_on_task
            if etk_stats:
                gaze_data_round = gaze_data.loc[(gaze_data["System Timestamp"] > row.StartDeviceTime) & (gaze_data["System Timestamp"] < row.EndDeviceTime), ["Event", "Device Timestamp"]]
                gaze_data_round = gaze_data_round.query("Event != 'Saccade'")
                results[row["Round"]]["FixationCount"] = len(gaze_data_round.Event.unique())
                results[row["Round"]]["MeanFixationDuration"] = gaze_data_round.groupby("Event").agg({"Device Timestamp": lambda x: x.iloc[-1] - x.iloc[0]}).mean()["Device Timestamp"]
        self.summarized_final_rounds[condition] = results
        return results
    
    def get_final_rounds(self, condition):
        roundData = self[condition]["RoundData"].copy()
        paras = Recording.parameter[condition]
        for p in paras:
            change_idx = (roundData[p] != roundData[p].shift(fill_value=roundData[p].loc[0])).cumsum()
            roundData[f"{p}_final"] = change_idx == change_idx.max()
        vali = roundData[[f"{p}_final" for p in paras]].all(axis=1)
        return self[condition]["RoundData"].loc[vali & ~roundData.EarlyStop]

    def calc_events(self, condition, th_dispersion=1, min_fixation_duration=50, max_fixation_duration=300, force_calculate=False):
        if "Event" in self[condition]["Gaze"].columns and not force_calculate:
            return
        vali = (self[condition]["Gaze120"].Message=="gaze sample") & self[condition]["Gaze120"].isValid
        gvs = self[condition]["Gaze120"].loc[vali, ["Local Gaze Direction %s" % x for x in ["X", "Y", "Z"]]].to_numpy()
        t = self[condition]["Gaze120"].loc[vali, "Device Timestamp"].to_numpy()
        fixations = utils.event_detection(gvs, t, th_dispersion, max_fixation_duration, min_fixation_duration)
        self[condition]["Gaze"]["Event"] = "Saccade"
        self[condition]["Gaze120"]["Event"] = "Saccade"
        for i, fix in enumerate(fixations):
            self[condition]["Gaze"].loc[(self[condition]["Gaze"]["Device Timestamp"] >= fix["StartTimestamp"]) & (self[condition]["Gaze"]["Device Timestamp"] <= fix["EndTimestamp"]), "Event"] = f"Fixation_{i}"
            self[condition]["Gaze120"].loc[(self[condition]["Gaze120"]["Device Timestamp"] >= fix["StartTimestamp"]) & (self[condition]["Gaze120"]["Device Timestamp"] <= fix["EndTimestamp"]), "Event"] = f"Fixation_{i}"
        self[condition]["Gaze"].to_csv(os.path.join(self.path, condition, "gaze_event.csv"), sep="\t", decimal=",", index=False)
        self[condition]["Gaze120"].to_csv(os.path.join(self.path, condition, "gaze_120fps_event.csv"), sep="\t", decimal=",", index=False)
        

    def calc_events_all_conditions(self, th_dispersion=1, min_fixation_duration=50, max_fixation_duration=300):
        for cond in tqdm(Recording.conditions, desc=f"Calc events of Pat {self.name}"):
            self.calc_events(cond, th_dispersion, min_fixation_duration, max_fixation_duration)

    

class Recordings:
    def __init__(self, path):
        pats = [x for x in os.listdir(path) if "." not in x]
        self.recs = [Recording(os.path.join(path, x)) for x in tqdm(pats)]
        self.pat_data = pd.read_csv(os.path.join(path, "pat_data.csv"), sep="\t", decimal=",", index_col=0)
        self._update_rec_data()
    
    def __len__(self):
        return len(self.recs)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.recs[index]
        elif isinstance(index, str):
            for rec in self:
                if rec.name==index:
                    return rec
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self[x] for x in index]
        else:
            raise TypeError("Invalid Argument Type")

    def _update_rec_data(self):
        for rec in self:
            name = rec.name
            if name in self.pat_data.index:
                rec.age, rec.gender, rec.vr_exp, rec.et_exp, rec.pref_method, rec.visual_aid = self.pat_data.loc[name, ["Age", "Gender", "VRExperience", "ETExperience", "PreferedMethod", "VisualAid"]]
            else:
                print(f"[WARNING] {name} not in pat_data.")

    def get_summarized_rounds(self, condition, gender=False):
        results = dict()
        for rec in self:
            if gender:
                if rec.gender != gender:
                    continue
            results[rec.name] = rec.summarize_rounds(condition)
        return results

    def calc_events(self, th_dispersion=1, min_fixation_duration=50, max_fixation_duration=300, force_calculation=False):
        for rec in self:
            for cond in Recording.conditions:
                rec.calc_events(cond, th_dispersion, min_fixation_duration, max_fixation_duration, force_calculate=force_calculation)

    def get_answers(self, condition, gender=False):
        answers = dict()
        for rec in self:
            if gender:
                if rec.gender != gender:
                    continue
            answers[rec.name] = dict()
            for index, row in rec["answers"].iterrows():
                answers[rec.name][index] = row[condition]
        return pd.DataFrame(answers).transpose()
    
    def get_all_answers(self, use_z_score=False, gender=False):
        questions = list(self[0]["answers"].index)
        answers = {x: {y: list() for y in questions} for x in Recording.conditions}
        for rec in self:
            if gender:
                if rec.gender != gender:
                    continue
            if use_z_score:
                rec_answers = rec["answers"]
                rec_answers = (rec_answers - rec_answers.loc[questions[:-1]].mean())/rec_answers.loc[questions[:-1]].std()
            else:
                rec_answers = rec["answers"].to_dict()
            for cond in Recording.conditions:
                for q in questions:
                    answers[cond][q].append(rec_answers[cond][q])
        return answers
    
    def get_final_parameters(self, condition, gender=False):
        paras = {x: list() for x in Recording.parameter[condition]}
        paras["Participant"] = list()
        for rec in self:
            if gender:
                if rec.gender != gender:
                    continue
            final_rounds_paras = rec.get_final_rounds(condition).loc[:, Recording.parameter[condition]]
            paras["Participant"].append(rec.name)
            for p in Recording.parameter[condition]:
                para = final_rounds_paras[p].to_numpy()[0]
                if type(para)==float: para=round(para,2)
                paras[p].append(para)
        return paras



if __name__=="__main__":
    import time
    path = "Data/TestRecordings"
    user = "ALena"
    # rec = Recording(os.path.join(path, user))
    recs = Recordings(path)
    print(recs.get_final_parameters("smoothPursuit"))

    

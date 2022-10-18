from sklearn import tree  # 결정 트리 알고리즘 라이브러리
import graphviz  # 결정 트리 그리는 라이브러리
import pandas as pd
from .data_util import Q_dict


def get_graph(model: tree.DecisionTreeClassifier):
    """학습된 결정 트리를 그림으로 그리는 함수

    Args:
        model (sklearn.tree.DecisionTreeClassifier): 학습이 끝난 결정 트리

    Returns:
        graphviz.files.Source: 결정 트리 그림
    """
    # 결정 트리를 graphviz로 그림
    dot_data = tree.export_graphviz(
                                    model,  # 그림을 그리기 위해 사용되는 결정 트리 모델
                                    out_file=None,  # 파일로 저장 안함
                                    feature_names=model.feature_names_in_,  # feature 이름 출력
                                    class_names=model.classes_,  # class 이름 출력
                                    filled=True,  # 클래스 별로 색깔 칠함
                                    rounded=True  # 테두리 둥글게 함
                                    )  
    graph = graphviz.Source(dot_data)
    return graph


def plot_feature_importance(model: tree.DecisionTreeClassifier):
    """학습이 끝난 결정 트리에서 feature 중요도를 바 그래프로 시각화하는 함수

    Args:
        model (sklearn.tree.DecisionTreeClassifier): 학습이 끝난 결정 트리
    """
    # 학습이 끝난 결정 트리에서 feature 중요도 정보 획득
    feat_importances = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_, columns=["Importance"])
    # 중요도 순으로 정렬
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
    # 바 그래프로 그림
    feat_importances.plot(kind='bar', figsize=(20,10))


def get_feature_importance_table(model: tree.DecisionTreeClassifier):
    """학습된 결정 트리에서 중요한 feature 순으로 정렬된 표를 얻는 함수

    Args:
        model (tree.DecisionTreeClassifier): 학습이 끝난 결정 트리

    Returns:
        _type_: feature 중요도 분석 표
    """
    # 학습이 끝난 결정 트리에서 feature 중요도 정보 획득
    feat_importances = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_, columns=["Importance"])
    # 중요도 순으로 정렬
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
    # 질문 항목에 설명 붙임
    feat_importances['desc'] = feat_importances.index.map(lambda x: Q_dict[x])
    return feat_importances
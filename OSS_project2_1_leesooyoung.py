import pandas as pd

# dataset 읽어 들이기 
data = pd.read_csv('2019_kbo_for_kaggle_v2.csv', encoding='ANSI')

print("Project #2-1 Data analysis with pandas\n")
print("2-1-(1)")
print("Print the top 10 players in H, avg, HR, OBP for each year from 2015 to 2018.\n")

# 규정타석을 기준으로 해당 년도의 각 지표 상위 10명 선수 출력(rank 써서 동점자 처리)
def get_top_players_with_rank(data, year, stat, required_pa):
    year_data = data[(data['year'] == year) & (data['PA'] >= required_pa)]
    top_players = year_data.sort_values(by=stat, ascending=False)
    top_players['rank'] = top_players[stat].rank(method='min', ascending=False)
    top_10_players = top_players[top_players['rank'] <= 10][['rank', 'batter_name', stat]]

    # Hits(H)와 Homerun(HR)은 정수 타입이므로 astype(int)로 정수 변환 
    if stat in ['H', 'HR']:
        top_10_players[stat] = top_10_players[stat].astype(int)

    # 결과 출력 
    print(f"Top players in {stat} ({year}):")
    print(top_10_players)
    print("\n" + "="*50 + "\n")

# 해당 년도와 각 지표를 나타내는 리스트 선언 
years = [2015, 2016, 2017, 2018]
stats = ['H', 'avg', 'HR', 'OBP']

# KBO 규정타석 = 전체 경기 수 (144경기) * 3.1 
required_pa = 144 * 3.1

# 2015-2018년 결과 출력 
for year in years:
    for stat in stats:
        get_top_players_with_rank(data, year, stat, required_pa)
        
# 2-1-(2)
# 2018년 가장 높은 WAR를 기록한 선수 출력하는 함수 
print("2-1-(2)")
def get_highest_war_by_position_2018(data, year):
    # 2018년 데이터 필터링 
    data_2018 = data[data['year'] == year]

    positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    
    # 각 포지션별 가장 높은 war를 기록한 선수를 찾는 함수 
    highest_war_players = {}
    for position in positions:
        players_in_position = data_2018[data_2018['cp'] == position] # 현재 포지션(cp) 기준
        if not players_in_position.empty:
            highest_war_player = players_in_position.loc[players_in_position['war'].idxmax()] #
            highest_war_players[position] = highest_war_player[['batter_name', 'war']]
    
    return highest_war_players

# 2018년 포지션별 가장 높은 war에 대한 정보 기록 Get the highest WAR by position for 2018
highest_war_by_position_2018 = get_highest_war_by_position_2018(data, 2018)

# 2018년 포션별 가장 높은 war를 기록한 선수 출력 
print("Print the player with the highest war by position (cp) in 2018.\n")
for position, player_info in highest_war_by_position_2018.items():
    # 소수점 둘째 자리 이후 숫자 버리기 위해 반올림 
    war_rounded = round(player_info['war'], 2)
    print(f"{position}: {player_info['batter_name']} - WAR: {war_rounded}")
print("\n" + "="*50 + "\n")

# 2-1-(3)
print("2-1-(3)")
print("Print which has the highest correlation with salary among R, H, HR, RBI, SB, war, avg, OBP and SLG.\n")
# 연봉과 가장 높은 상관관계를 가지는 통계치 찾는 함수 
def calculate_highest_corr_with_salary(data, required_pa):
    # 규정타석 충족한 선수 필터링 
    players = data[data['PA'] >= required_pa]

    # 상관관계 분석과 관련된 칼럼 선택
    columns_for_corr = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']

    # 상관계수 행렬 계산 
    corr_matrix = players[columns_for_corr].corr() 

    # 상관관계 분석 위해 salary 자신과의 상관관계 제거해 상관관계 값 추출 
    corr_with_salary = corr_matrix['salary'].drop('salary')

    # 헤당 시즌 연봉과 가장 높은 상관관계를 가진 통계 찾기 
    highest_corr_stat = corr_with_salary.idxmax()
    highest_corr_value = corr_with_salary.max()

    return highest_corr_stat, highest_corr_value

# 규정타석 구하기 
required_pa = 144 * 3.1

# 규정타석 충족한 선수 기준으로 연봉과 가장 높은 상관관계 가지는 stat 구하기 
highest_corr_stat, highest_corr_value = calculate_highest_corr_with_salary(data, required_pa)

# 결과 출력 
print(f"The stat with the highest correlation with salary is ")
print(f"{highest_corr_stat} (Correlation coefficient: {highest_corr_value:.2f})")
print("\n"*2)
print("End")


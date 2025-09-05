# label_maps.py
# 80개 음료 라벨 → 5그룹(s,c,h,d,pass) → 패킷(s/c/h/d or None)

from typing import Optional

# === 너가 올린 매핑 (필요시 자유롭게 수정) ===
BEVERAGE_LABEL_TO_GRP5 = {
    # --- Coca-Cola Korea (s)
    "CocaCola1-5L":"s","CocaCola250ML":"s","CocaCola2L":"s","CocaCola350ML":"s","CocaCola355ML":"s",
    "CocaColaOriginal300ML":"s","CocaColaZero250ML":"s","ZeroCocaCola215ML":"s",
    "FantaGrape250ML":"s","FantaOrage1-5L":"s","FantaOrange185ML":"s","FantaOrange215ML":"s","FantaPineapple1-5L":"s",
    "Sprite215ML":"s",
    "MinitmaidApple175ml":"s","MinitmaidMango175ML":"s","MinitmaidOrange1-5L":"s",
    "MinitmaidOriginalOrange1-5L":"s","MinitmaidTomato1-5L":"s",
    "MonsterEnergyPipelinePunch355ML":"s","Toreta1-5L":"s",

    # --- Lotte Chilsung (d)
    "PepsiCola1-5L":"d","PepsiCola600ML":"d","MountainDew355ML":"d",
    "Milkis1-5L":"d","Milkis250ML":"d","Milkis500ML":"d",
    "Let-sBe":"d","Let-sBe175ML":"d",
    "GalBaeCider210ML":"d","GalBaeCider238ML":"d","GalBaeCider355ML":"d","GalBaeCider500ML":"d",
    "CantataAmericano200ML":"d","CantataCafeMocha250ML":"d","CantataPremiumLatte":"d",
    "CantataSweetAmericano175ML":"d","CantataSweetAmericano275ML":"d",
    "CocoPalmGrape28ML":"d",
    "Welch-sGrape355ml":"d",
    "SunkistGrapeFruitSoda350ML":"d",

    # --- Haitai htb (h)
    "Coolpis":"h","CoolpisOriginalPeach":"h","CoolpisOriginalPineapple":"h","BirakSikhye":"h",

    # --- Dongwon F&B (c)
    "GayaAloeFarm1-5L":"c","GayaCarrotFarm1-5L":"c","GayaJejuGamgyulFarm1-5L":"c",
    "GayaTomatoFarm1-5L":"c","GayaTomatoFarm180ML":"c",
    "DelmontMango":"c","DelmontMango1-5L":"c",
    "JayeonunGamgyul1-5L":"c","JayeonunTomato500ML":"c",

    # --- pass (기타/불확실)
    "2perPeach350ML":"d",  # 2%는 롯데. d로 보낼거면 d, 애매하면 'pass'
    "AchimHaetsal180ML":"pass",
    "BosungNokcha":"pass",
    "CacaoNipsTea500ML":"pass",
    "CornSilkTea500ML":"pass",
    "Fita500Glass180ML":"pass",
    "FrenchCafeMildCoffee175ML":"pass",
    "GreenHoneyMaesil180ML":"pass",
    "GroundPear1L":"pass","GroundPear238ML":"pass","GroundPear340ML":"pass",
    "HaneulBori500ML":"pass",
    "HeotgaeCha1L":"pass","ConditionHeotgae100ML":"pass","ConditionHeotgaesoo340ML":"pass",
    "HiteTonicWater300ML":"pass",
    "MaeilSunupGreenFruitVegSalad":"pass","MaeilSunupGreenFruitVegSalad200ML":"pass",
    "MieroFiber210ML":"pass",
    "PocariSweat":"pass",
    "SeoulMilkAchimeJuiceApple210ML":"pass",
    "Sun-sMateTea500ML":"pass",
    "SunkistFamilyOrange180ML":"pass","SunkistFreshAloe1L":"pass","SunkistShineOnTheBeach340ML":"pass",
}

# 그룹→패킷 문자
_GROUP_TO_PACKET = {"s": "s", "c": "c", "h": "h", "d": "d"}

def beverage_label_to_packet(label: str) -> Optional[str]:
    """
    80라벨 중 하나인 label을 받아 s/c/h/d/None 중 하나를 반환.
    - 'pass' 또는 매핑없음 → None (패킷 전송 안 함)
    """
    grp = BEVERAGE_LABEL_TO_GRP5.get(label)
    return _GROUP_TO_PACKET.get(grp)  # pass 또는 None이면 그대로 None

# === (선택) 커버리지 체크: data.yaml과 매핑 딕셔너리의 불일치 출력 ===
def check_coverage(yaml_names: list[str]) -> None:
    missing = [n for n in yaml_names if n not in BEVERAGE_LABEL_TO_GRP5]
    if missing:
        print(f"[WARN] 매핑 없는 라벨 {len(missing)}개:", missing)
    else:
        print("[OK] 모든 라벨이 매핑되어 있습니다.")

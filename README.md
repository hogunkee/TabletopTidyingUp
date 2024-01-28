
## 기본 세팅
---

git clone --recursive [https://github.com/hogunkee/TabletopTidyingUp.git](https://github.com/hogunkee/TabletopTidyingUp.git)

python 3.8, ubuntu 20.04 (or 18.04)

pip install -r requirements.txt


## template collect
---

이미지에서 마우스 클릭으로 위치 선택

가능한 actions **(action들은 cv2 window에서 입력해야함.)**

- a : add object
    - 추가할 object category명, rotation을 terminal에서 입력. position은 마우스 클릭한 위치
    - 잘못 입력했다면 object category 입력시 q를 입력(terminal에서)하면 탈출 가능.
- d : delete object
    - 마우스 클릭한 위치에 object가 있으면, 그 object를 delete
- s : save template
    - 물체를 내가 만든 template대로 load해줌.
    - load한 template이 내가 의도한것과 같은지, 그리고 scene graph가 알맞게 만들어졌는지 확인 ⇒ 그렇다면 저장. (y/n 는 cv2 window에서 입력)
        - 내가 의도한것과 다르다면 강제로 scene을 reset해버리니 주의
    - 저장할때 template 번호를 terminal에 입력해서 저장.
- r : reset scene
    - scene을 reset함.
- q : quit
    - 현재 제작중인 scene에서 탈출.
- m : move object
    - 마우스로 물체를 선택했을때만 작동.
    - ‘w’, ‘a’, ‘s’, ‘d’, 로 물체를 이동. (0.03씩.)
    - ‘[’, ‘]’로 물체 회전. ([ : 반시계방향, ] : 시계방향으로 30도씩)
    - 물체 이동할때 테이블에 붙어있었던 물체여도 공중에 띄워버리니 참고
    - 끝내고 싶으면 q를 cv2 window에서 입력
- enter : simulate
    - 시뮬레이션을 잠깐 돌림.
    - 저장할때는 모든 물체들 다 테이블에 붙어있는 상태로 만들도록 시뮬레이션 돌려서 저장.

**마우스 클릭을 하지 않으면 이전에 마지막으로 클릭했던 마우스 위치가 남아서 물체가 선택될 수 있으니 주의!**


## template 수집시 주의사항
---

1. 테이블에 놓여진 물체들끼리는 서로 붙어있지 않도록 수집. (실제로 manipulation할때도 붙어있으면 집기가 힘들고, scene graph도 이상하게 만들어질 확률이 높음.)
2. 다만 다른 물체 위에 올라가는 경우는 가능한데, 이 경우에는 위에 있는 물체가 완전히 아래있는 물체 위에 올라가있도록 수집. (위에 있는 물체가 테이블이랑은 맞닿아 있지 않고 아래 있는 물체랑만 닿아있도록)

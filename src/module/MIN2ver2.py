"""
最小二乗法による円の検出を行う関数

main: MIN2_ignore_sunspots()

一度検出した近似円の内側にある点のうち、近似円の外側にある点だけから近似した円から
limb_wigth*(2/3) の範囲にないものは黒点とみなします。
"""

import pathlib
from pprint import pformat, pprint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Circle

version = "MIN2 v2.4.0"  # 黒点リトライ座標・iter_cycles=0・グローバル依存・描画まわりの修正

# spot の 4 番目の要素(走査方向)
DIR_X_LINE = 0  # 横線(yが固定)をx方向に走査して得た点
DIR_Y_LINE = 1  # 縦線(xが固定)をy方向に走査して得た点

# 旧コードとの互換用グローバル(img_inst="GLOBAL" を使う外部コード向け)。
# このファイル内の処理は、これらに頼らず引数で値を受け渡す。
img = None
divnum = None
height = 0
width = 0

_NO_IMAGE_MSG = (
    "画像が指定されていません。imgを渡す、グローバル変数imgを設定する、"
    "または画像のパスを引数に追加してください。"
)


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------
def _to_gray(image: np.ndarray) -> np.ndarray:
    """2次元のグレースケール画像に揃える(カラー/BGRA画像が来ても動くように)。"""
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        ch = image.shape[2]
        if ch == 1:
            return image[:, :, 0]
        if ch == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if ch == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"対応していない画像形状です: shape={image.shape}")


def _resolve_image(
    img_inst: str | np.ndarray, img_path: pathlib.Path | str | None = None
) -> np.ndarray:
    """img_inst("GLOBAL" / "PATH" / 画像配列)から2次元グレースケール画像を得る。"""
    if isinstance(img_inst, str):
        if img_inst == "GLOBAL":
            image = globals().get("img")
            if image is None:
                raise ValueError(_NO_IMAGE_MSG)
        elif img_inst == "PATH":
            if img_path is None or str(img_path) == "":
                raise ValueError(_NO_IMAGE_MSG)
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"画像の読み込みに失敗しました: {img_path}")
        else:
            raise ValueError(f"Unknown instructions = {img_inst}")
    else:
        image = img_inst
    if image is None:
        raise ValueError(_NO_IMAGE_MSG)
    return _to_gray(image)


def _resolve_divnum(n: int | None) -> int:
    if n is None:
        n = globals().get("divnum")
    if n is None:
        raise ValueError("分割数 n が指定されていません。")
    return int(n)


def _draw_circle(
    ax: plt.Axes,  # pyright: ignore[reportPrivateImportUsage]
    cir_stat: tuple[float, ...] | list[float] | np.ndarray | bool | None,
    color: str,
) -> None:
    if cir_stat is None or isinstance(cir_stat, bool):
        return
    cx, cy, R = cir_stat[:3]
    ax.add_patch(Circle((cx, cy), R, fill=False, color=color, linewidth=2))


def _draw_division_lines(
    ax: plt.Axes,  # pyright: ignore[reportPrivateImportUsage]
    width: int,
    height: int,
    n: int,
    alpha: float,  # pyright: ignore[reportPrivateImportUsage]
) -> None:
    """cut_and_sampling が実際に走査する位置(i=1..n-1)にだけ分割線を引く。"""
    for i in range(1, n):
        ax.axvline(width * i // n, color="white", linestyle="--", alpha=alpha)
        ax.axhline(height * i // n, color="white", linestyle="--", alpha=alpha)


# ---------------------------------------------------------------------------
# 検出処理
# ---------------------------------------------------------------------------
def cut_and_sampling(
    img_inst: str | np.ndarray, sun_threshold: float, n: int | None = None
) -> list[list[float]]:
    """画像を分割線で走査し、輝度の変化（微分値）から太陽の縁（エッジ）に相当する点の座標をサンプリングする。

    Args:
        img_inst (Union[str, np.ndarray]): 画像の指示。"GLOBAL"ならglobal変数のimgを取得する。もしくは読み込んだ画像。
        sun_threshold (Union[int, float]): 太陽像とみなす明るさのしきい値。これ以下の直線は処理をスキップする。
        n (int | None): 分割数。Noneならglobal変数divnumを使う。

    Returns
    -------
        list[list[float]]: サンプリングされた縁の点 [x, y, grad_val, direc] のリスト。
            direc は DIR_X_LINE(0:横線走査) / DIR_Y_LINE(1:縦線走査)。
    """
    image = _resolve_image(img_inst)
    n = _resolve_divnum(n)
    h, w = image.shape[:2]

    spots: list[list[float]] = []
    for line_xy in ("x_line", "y_line"):  # x_lineは横線、y_lineは縦線
        for i in range(1, n):
            if line_xy == "x_line":
                place = h * i // n
                line = image[place, :].astype(float)
            else:
                place = w * i // n
                line = image[:, place].astype(float)
            if np.max(line) <= sun_threshold:  # 太陽像上を通るか
                continue
            grad_t = np.diff(line)  # 一回微分
            max_idx = int(np.argmax(grad_t))
            min_idx = int(np.argmin(grad_t))
            if line_xy == "x_line":
                spots.append([max_idx, place, grad_t[max_idx], DIR_X_LINE])
                spots.append([min_idx, place, grad_t[min_idx], DIR_X_LINE])
            else:
                spots.append([place, max_idx, grad_t[max_idx], DIR_Y_LINE])
                spots.append([place, min_idx, grad_t[min_idx], DIR_Y_LINE])
    return spots


def fit_circle(
    spots: list[list[float]] | np.ndarray,
    show: bool = False,
    img_inst: str | np.ndarray = "GLOBAL",
    n: int | None = None,
) -> list[float]:
    """与えられた縁の点の座標群から、最小二乗法を用いて近似円の中心座標と半径を計算する。

    Args:
        spots: 縁の点の座標 [x, y, ...] を格納した二次元配列、またはNumPy配列。
        show (bool): 例外発生時にshow_circleによる描画を行うか
        img_inst: 例外時の描画に使う画像(既定は"GLOBAL")。
        n (int | None): 例外時の描画に使う分割数。

    Raises
    ------
        ValueError: 3点未満、または円を確定できない場合。

    Returns
    -------
        list[float]: [cx, cy, R]
    """
    if len(spots) < 3:
        print("[ERROR]:点が3点未満のため、円を作成できません。too little spots")
        if show:
            show_circle(img_inst=img_inst, spots=spots, cir_stat=False, n=n)
        raise ValueError("点不足")
    x = np.array([s[0] for s in spots], dtype=float)
    y = np.array([s[1] for s in spots], dtype=float)
    mat_A = np.c_[x, y, np.ones(len(x))]
    vec_B = -(x**2 + y**2)
    res, _, _, _ = np.linalg.lstsq(mat_A, vec_B, rcond=None)
    A, B, C = res
    cx = -A / 2
    cy = -B / 2
    radicand = cx**2 + cy**2 - C
    if radicand <= 0:  # 点が一直線上に並ぶなどで円にならない
        raise ValueError("円を確定できません(半径の2乗が正になりません)")
    return [float(cx), float(cy), float(np.sqrt(radicand))]


def retry_edge_on_spot(
    spot: list[float],
    cir_stat: tuple[tuple[float, float], float],
    img_inst: np.ndarray | str = "GLOBAL",
    sun_threshold: int = 50,
) -> None | list:
    """黒点を誤検知したときに、近似円の外側で縁を探し直す関数。

    spot と同じ走査線上で、近似円の縁より外側の範囲だけを再走査する。
    見つからなければ None。
    """
    image = _resolve_image(img_inst)
    spot_x, spot_y, grad_val, direc = spot
    (cx, cy), r = cir_stat
    direc = int(direc)
    h, w = image.shape[:2]

    if direc == DIR_X_LINE:
        perpendic = abs(spot_y - cy)
    elif direc == DIR_Y_LINE:
        perpendic = abs(spot_x - cx)
    else:
        raise ValueError(f"Unknown direction in spot stat (expected 0 or 1): {direc}")

    variation = np.sqrt(max(r * r - perpendic * perpendic, 0.0))

    if direc == DIR_X_LINE:
        if grad_val < 0:  # 明→暗: 右側の縁
            x0 = int(max(0, cx + variation))
            x1 = w
        else:  # 暗→明: 左側の縁
            x0 = 0
            x1 = int(min(w, cx - variation))
        y0 = int(np.clip(spot_y, 0, h - 1))
        if x1 <= x0:
            return None
        line = image[y0, x0:x1].astype(float)
    else:
        if grad_val < 0:  # 明→暗: 下側の縁
            y0 = int(max(0, cy + variation))
            y1 = h
        else:  # 暗→明: 上側の縁
            y0 = 0
            y1 = int(min(h, cy - variation))
        x0 = int(np.clip(spot_x, 0, w - 1))
        if y1 <= y0:
            return None
        line = image[y0:y1, x0].astype(float)

    # 微分するには2要素以上必要(1要素だと argmax が空配列で例外になる)
    if line.size < 2 or np.max(line) <= sun_threshold:
        return None

    grad_t = np.diff(line)
    rel_idx = int(np.argmax(grad_t)) if grad_val > 0 else int(np.argmin(grad_t))
    grad_at_edge = float(grad_t[rel_idx])

    # 走査方向に沿った相対位置を画像座標に戻す。
    # 横線走査ならxが、縦線走査ならyが動く
    if direc == DIR_X_LINE:
        return [x0 + rel_idx, y0, grad_at_edge, direc]
    return [x0, y0 + rel_idx, grad_at_edge, direc]


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------
def show_circle(
    img_inst: str | np.ndarray,
    spots: list[list[float]] | np.ndarray | None = None,
    cir_stat: tuple[float, float, float] | list[float] | bool = False,
    img_path: pathlib.Path | str | None = None,
    fig_info: dict[str, str] | None = None,
    iteration_count: int | str = 1,
    is_last: bool = False,
    simple: bool = False,
    n: int | None = None,
) -> None:
    """画像上に分割線、サンプリングされた縁の点、近似円を描画し、各エッジ点付近の明るさと微分の2軸グラフを右側に並べて表示する。

    Args:
        img_inst: "GLOBAL" / "PATH"(img_pathから読み込み) / 読み込んだ画像。
        spots: 描画する縁の点 [x, y, grad, direc] のリスト。
        cir_stat: 近似円 [cx, cy, R]。描画しない場合は False。
        img_path: 画像のファイルパス。
        fig_info: 画像内にテキスト表示するメタデータ。
        iteration_count: 現在の反復回数。
        is_last: Trueなら反復回数の代わりに "Last" と表示。
        simple: Trueならシンプルな表示(ポスター図用)のみを行う。
        n: 分割数。Noneならglobal変数divnumを使う。
    """
    if simple:
        show_circle_simple(
            img_inst=img_inst,
            spots=spots,
            cir_stat=cir_stat,
            img_path=img_path,
            iteration_count=iteration_count,
            is_last=is_last,
            n=n,
        )
        return

    image = _resolve_image(img_inst, img_path)
    n = _resolve_divnum(n)
    h, w = image.shape[:2]
    if spots is None:
        spots = []

    # 点がない場合は画像のみ表示
    if len(spots) == 0:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="magma")
        _draw_circle(ax, cir_stat, "orange")
        plt.show()
        return

    num_spots = len(spots)
    cols = 5  # 右側に並べる小グラフの列数
    rows = (num_spots - 1) // cols + 1

    # FigureとGridSpecの作成（左側3列分をメイン画像、右側を小グラフ群に）
    fig = plt.figure(figsize=(15, max(6, rows * 2)))
    gs = gridspec.GridSpec(rows, cols + 3, figure=fig)

    iter_text = "Last" if is_last else str(iteration_count)
    img_name = pathlib.Path(img_path).name if img_path else "Unknown"
    fig.suptitle(f"{img_name}  |  Iteration: {iter_text}", fontsize=16, fontweight="bold")

    # メイン画像の描画
    ax_main = fig.add_subplot(gs[:, :3])
    ax_main.set_title(f"{img_path}", fontsize=9, color="gray", loc="left", pad=10)
    if fig_info:
        # データ座標(0.05, 0.05)だと画像の左上端の外に出てしまうため、軸座標で指定する
        ax_main.text(
            0.05,
            0.05,
            pformat(fig_info, indent=2, width=40),
            ha="left",
            va="bottom",
            fontsize=12,
            transform=ax_main.transAxes,
        )
    ax_main.imshow(image, cmap="magma")
    _draw_circle(ax_main, cir_stat, "orange")

    x = np.array([s[0] for s in spots], dtype=float)
    y = np.array([s[1] for s in spots], dtype=float)
    ax_main.scatter(x, y, color="red", label="Edges", s=50)

    # 座標ラベルと対応関係のための番号を表示
    for idx, (xi, yi) in enumerate(zip(x, y, strict=False)):
        ax_main.text(
            xi,
            yi,
            f"#{idx + 1}",
            color="lime",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        ax_main.text(
            xi,
            yi,
            f"({xi:.0f}, {yi:.0f})",
            color="#8917fd",
            fontsize=8,
            ha="left",
            va="top",
        )

    _draw_division_lines(ax_main, w, h, n, alpha=0.3)
    ax_main.text(0.05, 0.9, f"n={n}", color="cyan", fontsize=10, transform=ax_main.transAxes)
    ax_main.legend()
    ax_main.axis("equal")

    # === 各エッジ点付近の小グラフを作成 ===
    window_size = 15  # 抽出する近傍のサイズ（前後15ピクセル）
    for idx, spot in enumerate(spots):
        xi = int(np.clip(round(float(spot[0])), 0, w - 1))
        yi = int(np.clip(round(float(spot[1])), 0, h - 1))

        # 走査方向は spot が持っている。座標が分割線上かどうかでの推測だと、
        # 縦線走査の点がたまたま横線のy座標と一致したときに取り違える。
        if len(spot) > 3:
            is_x_line = int(spot[3]) == DIR_X_LINE
        else:
            is_x_line = any(yi == h * i // n for i in range(1, n))

        if is_x_line:
            line_data = image[yi, :].astype(float)
            center_idx = xi
        else:
            line_data = image[:, xi].astype(float)
            center_idx = yi

        start = max(0, center_idx - window_size)
        end = min(len(line_data), center_idx + window_size + 1)
        vals = line_data[start:end]
        # np.diffは要素が1つ減るため、プロット用に末尾に0を追加して長さを合わせる
        grad_vals = np.append(np.diff(line_data), 0)[start:end]
        x_coords = np.arange(start, end) - center_idx

        ax_sub = fig.add_subplot(gs[idx // cols, 3 + idx % cols])
        ax_sub.set_title(f"#{idx + 1}", fontsize=10, color="black", fontweight="bold")

        color_bright = "tab:orange"
        ax_sub.plot(x_coords, vals, color=color_bright, linewidth=1.5)
        ax_sub.tick_params(axis="y", labelcolor=color_bright, labelsize=7)
        ax_sub.tick_params(axis="x", labelsize=7)
        ax_sub.grid(alpha=0.3)

        ax_sub_twin = ax_sub.twinx()
        color_diff = "tab:cyan"
        ax_sub_twin.plot(x_coords, grad_vals, color=color_diff, linewidth=1.5, linestyle="--")
        ax_sub_twin.tick_params(axis="y", labelcolor=color_diff, labelsize=7)

        ax_sub.axvline(0, color="red", linestyle="-", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.show()


def show_circle_simple(
    img_inst: str | np.ndarray,
    spots: list[list[float]] | np.ndarray | None = None,
    cir_stat: tuple[float, float, float] | list[float] | bool = False,
    img_path: pathlib.Path | str | None = None,
    iteration_count: int | str = 1,
    is_last: bool = False,
    markersize: int = 400,
    n: int | None = None,
) -> None:
    """画像上に分割線、サンプリングされた縁の点、近似円をシンプルに描画して表示する。

    Args:
        img_inst: "GLOBAL" / "PATH"(img_pathから読み込み) / 読み込んだ画像。
        spots: 描画する縁の点の座標リスト。
        cir_stat: 近似円のステータス [cx, cy, R]。描画しない場合は False。
        img_path: 画像のファイルパス。
        iteration_count: 現在の反復回数。
        is_last: Trueなら "Last" と表示。
        markersize: 縁の点のマーカーサイズ。
        n: 分割数。Noneならglobal変数divnumを使う。
    """
    image = _resolve_image(img_inst, img_path)
    n = _resolve_divnum(n)
    h, w = image.shape[:2]
    if spots is None:
        spots = []

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="viridis")
    _draw_circle(ax, cir_stat, "white")
    if len(spots) > 0:
        x = np.array([s[0] for s in spots], dtype=int)
        y = np.array([s[1] for s in spots], dtype=int)
        ax.scatter(
            x,
            y,
            color="red",
            label="Edges",
            s=markersize,
            linewidths=2,
            edgecolors="white",
        )

    img_name = pathlib.Path(img_path).name if img_path else "Unknown"
    iter_text = "Last" if is_last else str(iteration_count)
    fig.suptitle(f"{img_name}  |  Iteration: {iter_text}", fontsize=16, fontweight="bold")

    _draw_division_lines(ax, w, h, n, alpha=0.5)
    ax.text(0.05, 0.9, f"n={n}", color="cyan", fontsize=10, transform=ax.transAxes)
    if len(spots) > 0:
        ax.legend()
    ax.axis("equal")
    plt.show()


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def MIN2_ignore_sunspots(
    img_inst: np.ndarray | str = "PATH",
    n: int = 10,
    light_threshold: int = 50,
    limb_wigth: int = 24,
    iter_cycles: int = 2,
    show: bool = False,
    debug: bool = False,
    img_path: pathlib.Path | str | None = "",
    show_simple: bool = False,
) -> tuple[tuple[float, float], float]:
    """黒点（サンスポット）による影響を除外しながら、最小二乗法により太陽の最終的な近似円（中心と半径）を検出します。

    一度検出した近似円の外側にある点から再度円を近似し、その円の縁から limb_wigth*(2/3) より
    内側に離れている内側の点を黒点とみなして除外します。

    Args:
        img_inst: 入力画像（グレースケール/カラー）または読み込み指示("PATH"ならimg_pathを読み込む)。
        n (int): 画像格子の分割数(2以上)。デフォルトは 10 です。
        light_threshold (int): 太陽の明るさの基準しきい値。16bit画像では自動で256倍されます。
        limb_wigth (int): 太陽の縁の幅の基準値。デフォルトは 24 です。
        iter_cycles (int): 黒点排除の最大反復回数。複数の黒点に対応できます。0なら排除なし。デフォルトは2。
            黒点が見つからなくなった時点で早期終了します。
        show (bool): 最終的な検出結果の画像を表示するかどうか。
        debug (bool): 各ステップの描画(showもTrueのとき)やログを出力するかどうか。
        img_path: 処理する画像のファイルパス("PATH"指定時は必須。配列を渡した場合は表示用の名前として使うだけ)。
        show_simple (bool): 描画時に詳細なグラフを省いたシンプルな表示形式を使用するかどうか。

    Returns
    -------
        tuple[tuple[float, float], float]: ((cx, cy), r)
    """
    global divnum, img, height, width  # 外部コード互換のため値を残す(内部では使わない)

    if n < 2:
        raise ValueError("分割数 n は 2 以上にしてください。")

    path_obj = pathlib.Path(img_path) if img_path else None
    if isinstance(img_inst, str):
        if img_inst != "PATH":
            raise ValueError(f"Unknown instructions = {img_inst}")
        # パスの存在チェックは、実際にそれを読み込むときだけ行う
        if path_obj is None or not path_obj.is_file():
            raise ValueError(f"そのパスの画像は存在しません。: {img_path}")
        readed_img = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
        if readed_img is None:
            raise ValueError("画像の読み込みに失敗しました。")
    else:
        readed_img = img_inst

    image = _to_gray(readed_img)  # カラー画像でも2次元で扱えるようにする
    divnum, img = n, image
    height, width = image.shape[:2]

    if image.dtype == np.uint16:
        light_threshold = light_threshold * 256

    spots = cut_and_sampling(image, light_threshold, n)
    if debug:
        pprint(spots)
    cx, cy, r = fit_circle(spots, show, image, n)  # 一回目の円情報
    if debug:
        print(f"[INFO]:trial circle (cx,cy,r)={cx, cy, r}")
        if show:
            show_circle(
                img_inst=image,
                spots=spots,
                cir_stat=(cx, cy, r),
                img_path=path_obj,
                iteration_count=1,
                fig_info={"circle": "first trial"},
                simple=show_simple,
                n=n,
            )

    all_sunspots = []
    safe_points = list(spots)

    for cycle in range(1, iter_cycles + 1):
        if debug:
            print(f"iter: {cycle}")

        outside_spots = []
        inside_spots = []
        for point in safe_points:
            x, y = point[0], point[1]
            # intに丸めるとoutsideが極端に少なくなるので丸めない
            if np.hypot(x - cx, y - cy) > r:
                outside_spots.append(point)
            else:
                inside_spots.append(point)

        # inside_spotsのindexで黒点を管理するため、insideを先頭に並べる
        safe_points = inside_spots + outside_spots

        if len(outside_spots) < 3:
            print("[WARN]:外側の点が3点未満のため、黒点排除を打ち切ります。")
            break
        try:
            cxo, cyo, ro = fit_circle(np.array(outside_spots, dtype=float), show, image, n)
        except ValueError as e:
            print(f"[WARN]:外側の点から円を作れないため、黒点排除を打ち切ります。({e})")
            break
        if debug:
            print(f"[INFO]:outside circle (cx,cy,r)={cxo, cyo, ro}")
            if show:
                show_circle(
                    img_inst=image,
                    spots=outside_spots,
                    cir_stat=(cxo, cyo, ro),
                    img_path=path_obj,
                    iteration_count=cycle,
                    fig_info={"circle": "only outside points "},
                    simple=show_simple,
                    n=n,
                )
            print(f"    [INFO]:外側の点の数:{len(outside_spots)},全体の点の数:{len(spots)}")

        sunspots: list[int] = []
        retry_points: list[list] = []
        for i, point in enumerate(inside_spots):
            x, y = point[0], point[1]
            if (x - cxo) ** 2 > (y - cyo) ** 2:  # 円の左右(LR)側
                # 負になると np.sqrt が NaN を返し、比較が常に False になって黒点を見逃すので0で打ち止め
                min2far = np.sqrt(max(ro**2 - (y - cyo) ** 2, 0.0))
                gap = min2far - np.abs(cxo - x)
                tag = "LR"
            else:  # 円の上下(TB)側
                min2far = np.sqrt(max(ro**2 - (x - cxo) ** 2, 0.0))
                gap = min2far - np.abs(cyo - y)
                tag = "TB"
            if debug:
                print(f"    {i} x,y:{x, y} min2far:{min2far},gap:{gap}")

            if gap > limb_wigth * (2 / 3):
                sunspots.append(i)
                retry = retry_edge_on_spot(
                    spot=point,
                    cir_stat=((cx, cy), r),
                    img_inst=image,
                    sun_threshold=light_threshold,
                )
                if debug:
                    print(
                        f"    黒点を検出しました({tag})。再検討します。\n    {point}\n    → {retry}"
                    )
                if retry is not None:
                    retry_points.append(retry)

        for i in sorted(sunspots, reverse=True):
            all_sunspots.append(safe_points.pop(i))
        safe_points.extend(retry_points)

        if not sunspots:
            break
        if len(safe_points) < 3:
            print("[WARN]:黒点除外後の点が3点未満のため、直前の円を採用します。")
            break
        # 黒点とみなされない点だけで円を作成
        cx, cy, r = fit_circle(np.array(safe_points, dtype=float), show, image, n)

    if show:
        show_circle(
            img_inst=image,
            spots=safe_points,
            cir_stat=(cx, cy, r),
            img_path=path_obj,
            is_last=True,
            simple=show_simple,
            n=n,
        )
    return (cx, cy), r


def _collect_images(dirpath: str) -> list[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(p for p in pathlib.Path(dirpath).iterdir() if p.suffix.lower() in exts)


def main() -> None:
    """Run the main command-line process."""
    import argparse
    from time import perf_counter

    DIR_MODE_LIMB_WIDTH = 60

    parser = argparse.ArgumentParser(description="太陽像の近似円を検出する")
    parser.add_argument(
        "path", nargs="?", help="画像ファイルまたはフォルダ。省略時はダイアログで選択"
    )
    parser.add_argument("--dir", action="store_true", help="フォルダ内の全画像を処理する")
    args = parser.parse_args()

    path = args.path
    if path is None:
        import tkinter as tk
        from tkinter.filedialog import askdirectory, askopenfilename

        root = tk.Tk()
        root.withdraw()  # 空のTkウィンドウが出ないようにする
        if args.dir:
            path = askdirectory(title="フォルダを選択してください")
        else:
            path = askopenfilename(
                title="画像を選択してください",
                # セミコロン区切りの1文字列はWindows以外で効かないのでタプルで渡す
                filetypes=[("Image files", ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"))],
            )
        root.destroy()
        if not path:
            print("[INFO]:選択がキャンセルされました。")
            return

    if args.dir or pathlib.Path(path).is_dir():
        print(f"[INFO]:dir={path}")
        for file in _collect_images(path):
            try:
                (cx, cy), r = MIN2_ignore_sunspots(
                    "PATH",
                    show=True,
                    debug=True,
                    limb_wigth=DIR_MODE_LIMB_WIDTH,
                    img_path=file,
                )
            except ValueError as e:
                # 旧コードは読み込み失敗で break して残りを全部スキップしていた
                print(f"[ERROR]:{file}: {e}")
                continue
            print((float(cx), float(cy), float(r)))
    else:
        print(f"[INFO]:image={path}")
        start = perf_counter()
        result = MIN2_ignore_sunspots(
            "PATH", show=True, debug=True, img_path=path, show_simple=False
        )
        print(f"[INFO]:result{result}")
        print(f"[INFO]:process time :{perf_counter() - start} s")


if __name__ == "__main__":
    main()
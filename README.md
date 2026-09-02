# Sphere Pong Wars for M5Stack

[3D Sphere Pong Wars](https://github.com/K-Yama2010/3D_Sphere_Pong_Wars)を
M5Unifiedベースの単体スケッチへ移植したプロジェクトです。M5Stack StopWatchの
円形ディスプレイ、タッチ、A/Bボタンに対応します。

## 特徴

- 8×8×6面のクアッドスフィア
- 2ペア、合計4エージェント
- 開始座標と球面接線方向を毎ゲームランダム化
- ゼロ長ベクトルを安全に扱う原典準拠の正規化
- 黒と6種類のネオンカラー
- タッチドラッグによる慣性回転
- 原典と同じ背面遮蔽、暗色表示、球面に沿った四角形エージェント

## 操作

- Aボタン: 新しいゲーム
- Bボタン: ネオンカラー切替
- タッチドラッグ: 球体の回転方向と速度を変更

## ビルド

Arduino IDEでは `SpherePongWars.ino` を開き、ボードにM5Stack StopWatchを
指定します。Arduino CLIでは以下を使用できます。

```sh
arduino-cli compile --fqbn m5stack:esp32:m5stack_stopwatch .
arduino-cli upload --port COM7 --fqbn m5stack:esp32:m5stack_stopwatch .
```

原典および本プロジェクトのライセンスは `LICENSE` を参照してください。

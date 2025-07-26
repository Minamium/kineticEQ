program main
    ! モジュールを指定
    use sample_1d_advection_module
    implicit none
    ! パラメータ設定
    integer, parameter :: nx = 10 ! 物理空間分割数の定義

    ! 物理長と時間分割幅, シミュレート時間と流速の定義
    real, parameter :: Lx = 1.0, dt = 0.001, Time = 0.1, u = 0.4

    ! 濃度関数配列の定義
    real :: q(nx), q_new(nx)

    ! その他パラメータの定義と計算
    real :: dx
    integer :: step

    ! ユーティリティ変数定義
    integer :: i, t

    dx = 1.0 / nx
    step = int(Time / dt)

    ! パラメータ表示
    print *, "物理空間分割数=",nx
    print *, "物理空間分割幅=",dx
    print *, "時間分割幅=", dt
    print *, "ステップ数=", step

    ! 初期条件設定
    do i = 1, nx
        if ( (i*dx > 0.4) .and. (i*dx < 0.6) ) then
            q(i) = 0.5
        else
            q(i) = 0.0
        end if
    end do

    ! シミュレーション実行
    do t = 1, step
        ! モジュール呼び出し
        call advec_upwind_step(nx, dt, dx, u, q, q_new)

        ! 配列交換
        q = q_new

        ! 進捗表示
        if (mod(t, max(1, step / 10)) == 0) then
            print *, "Progress:", 100.0 * (real(t) / real(step)), "%"

            ! この時点での濃度関数の表示
            print *, q
        end if
    end do
end program
    

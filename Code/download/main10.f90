program main
    implicit none
    character(len=29) :: arg
    character(len=8) :: filelist
    integer :: status, io, NoF, N
    integer*2, parameter :: pix = 12000
    integer*2, dimension(pix, pix) :: data_all
    real, dimension(pix, pix) :: data_tbb
    real*4, dimension(5000) :: tbb
    integer*2, dimension(5000) :: cnt
    integer*2 :: i, j, index

    ! Lấy tham số từ dòng lệnh
    call getarg(1, arg)
    call getarg(2, filelist)

    ! Mở file dữ liệu
    open(21, file=arg, access='direct', status='old', iostat=io, recl=pix*pix*2)
    if (io /= 0) then
        print *, "Lỗi mở file dữ liệu: ", trim(arg)
        stop
    end if

    read(21, rec=1, iostat=io) ((data_all(i, j), i=1, pix), j=1, pix)
    close(21)

    if (io /= 0) then
        print *, "Lỗi đọc file dữ liệu!"
        stop
    end if

    ! Kiểm tra min/max của dữ liệu đầu vào
    print *, "Giá trị min/max của data_all:", minval(data_all), maxval(data_all)

    ! Mở file bảng TBB
    NoF = 5000
    open(20, file=filelist, status='old', access='sequential', form='formatted', iostat=io)
    if (io /= 0) then
        print *, "Lỗi mở file bảng TBB: ", trim(filelist)
        stop
    end if

    do N = 1, NoF
        read(20, *, iostat=io) cnt(N), tbb(N)
        if (io /= 0) exit
    end do
    close(20)

    ! Kiểm tra một số giá trị từ bảng TBB
    print *, "Tbb values (1st, last): ", tbb(1), tbb(NoF)

    ! Chuyển đổi CTT sang TBB
    do i = 1, pix
        do j = 1, pix
            index = data_all(i, j) + 1

            ! Kiểm tra chỉ số hợp lệ trước khi truy xuất mảng
            if (index >= 1 .and. index <= NoF) then
                data_tbb(i, j) = real(tbb(index))
            else
                data_tbb(i, j) = -999.0  ! Giá trị lỗi
            end if

            ! Debug dữ liệu cho 10 pixel đầu tiên
            if (i == 1 .and. j <= 10) then
                print *, "data_all(", i, j, ") =", data_all(i, j), " -> index =", index, " -> tbb =", data_tbb(i, j)
            end if
        end do
    end do

    ! Ghi dữ liệu ra file .dat
    open(10, file="grid10.dat", access='direct', recl=pix*pix*4, status='replace', iostat=io)
    if (io /= 0) then
        print *, "Lỗi mở file đầu ra!"
        stop
    end if

    write(10, rec=1, iostat=io) ((data_tbb(i, j), i=1, pix), j=1, pix)
    close(10)

    if (io /= 0) then
        print *, "Lỗi ghi file đầu ra!"
        stop
    end if

    print *, "Chuyển đổi hoàn tất! File đầu ra: grid10.dat"
end program

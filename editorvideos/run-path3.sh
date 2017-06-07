    php new.php path3.defs > inputter.dat

    rm capture; ln output/path3 -s capture
    rm capture/e_*.avi capture/inputter_*.avi
    
    #cd ~/qbexample/
    trap "cd -" SIGINT
    LD_LIBRARY_PATH="$(pwd)/dblib" \
    ./dosbox-works
    trap SIGINT
#    cd -

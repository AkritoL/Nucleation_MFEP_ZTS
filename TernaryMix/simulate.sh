DIMENSION=2d
#DIMENSION=3d
CASE=AsymmetricCase
#CASE=SymmetricCase
NUCLEATION_MODE=heterogeneous
#NUCLEATION_MODE=homogeneous
INTERFACE_WIDTH=0.012
#INTERFACE_WIDTH=0.024
GAMMA_AB=1e-2
GAMMA_BC=1e-2
GAMMA_CA=1.3e-2
TIME_STEP=5e-6

case $CASE in
    "SymmetricCase")
        OUTPUT_DIR=$DIMENSION/Output/$CASE/Data-InterfaceWidth-$INTERFACE_WIDTH/gamma_ab-$GAMMA_AB/$NUCLEATION_MODE
        ;;
    "AsymmetricCase")
        OUTPUT_DIR=$DIMENSION/Output/$CASE/Data-InterfaceWidth-$INTERFACE_WIDTH/gamma_ca-$GAMMA_CA/$NUCLEATION_MODE
        ;;
    *)
        echo "Invalid operation"
        ;;
esac

python $DIMENSION/main.py \
    --NUM_GPU 0 \
    --NUCLEATION_MODE $NUCLEATION_MODE \
    --CASE $CASE \
    --INTERFACE_WIDTH $INTERFACE_WIDTH \
    --GAMMA_AB $GAMMA_AB \
    --GAMMA_BC $GAMMA_BC \
    --GAMMA_CA $GAMMA_CA \
    --FINAL_RADIUS 0.1 \
    --STEPMAX 500000 \
    --TIME_STEP $TIME_STEP \
    --TIME_STEP_INCREMENT 0 \
    --GRID_NUMBER 256 \
    --NUM_STRING_IMAGES 50 \
    --OUTPUT_DIR $OUTPUT_DIR \
    --PLOT_ENERGY true \
    --PLOT_CONCENTRATION_RGB true \
    --SAVE_CONCENTRATION_DATA true \
    --SAVE_ENERGY_DATA true \
    --PRE_DATA_LOAD false \
    

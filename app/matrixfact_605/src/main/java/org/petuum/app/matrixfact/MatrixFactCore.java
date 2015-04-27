package org.petuum.app.matrixfact;

import org.petuum.app.matrixfact.Rating;
import org.petuum.app.matrixfact.LossRecorder;

import org.petuum.ps.PsTableGroup;
import org.petuum.ps.row.double_.DenseDoubleRow;
import org.petuum.ps.row.double_.DenseDoubleRowUpdate;
import org.petuum.ps.row.double_.DoubleRow;
import org.petuum.ps.row.double_.DoubleRowUpdate;
import org.petuum.ps.table.DoubleTable;
import org.petuum.ps.common.util.Timer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class MatrixFactCore {
  private static final Logger logger = LoggerFactory
      .getLogger(MatrixFactCore.class);

  // Perform a single SGD on a rating and update LTable and RTable
  // accordingly.
  public static void sgdOneRating(Rating r, double learningRate,
      DoubleTable LTable, DoubleTable RTable, int K, double lambda) {
    // TODO
    DoubleRow Li = LTable.get(r.userId);
    DoubleRow Rj = RTable.get(r.prodId);

    double Eij = r.rating - computeDotProduct(Li, Rj, K);
    double ni = Li.get(K);
    double nj = Rj.get(K);

    DoubleRowUpdate lUpdate = new DenseDoubleRowUpdate(K);
    DoubleRowUpdate rUpdate = new DenseDoubleRowUpdate(K);
    for (int i = 0; i < K; i++) {
      lUpdate.setUpdate(i,
          2 * learningRate * (Eij * Rj.get(i) - lambda * Li.get(i) / ni));
      rUpdate.setUpdate(i,
          2 * learningRate * (Eij * Li.get(i) - lambda * Rj.get(i) / nj));
    }

    LTable.batchInc(r.userId, lUpdate);
    RTable.batchInc(r.prodId, rUpdate);

  }

  private static double computeDotProduct(DoubleRow li, DoubleRow rj, int end) {
    double dotProduct = 0;
    for (int i = 0; i < end; i++) {
      dotProduct += li.get(i) * rj.get(i);
    }
    return dotProduct;
  }

  // Evaluate square loss on entries [elemBegin, elemEnd), and L2-loss on of
  // row [LRowBegin, LRowEnd) of LTable, [RRowBegin, RRowEnd) of Rtable.
  // Note the interval does not include LRowEnd and RRowEnd. Record the loss to
  // lossRecorder.
  public static void evaluateLoss(ArrayList<Rating> ratings, int ithEval,
      int elemBegin, int elemEnd, DoubleTable LTable, DoubleTable RTable,
      int LRowBegin, int LRowEnd, int RRowBegin, int RRowEnd,
      LossRecorder lossRecorder, int K, double lambda) {
//    double sqLoss = 0;
//    double totalLoss = 0;

    for (int i = elemBegin; i < elemEnd; i++) {
      Rating r = ratings.get(i);
      DoubleRow Li = LTable.get(r.userId);
      DoubleRow Rj = RTable.get(r.prodId);
      double sqLossAdd = Math.pow(r.rating - computeDotProduct(Li, Rj, K), 2);
 
      lossRecorder.incLoss(ithEval, "SquareLoss", sqLossAdd);//could be commented
      lossRecorder.incLoss(ithEval, "FullLoss", sqLossAdd);//could be commented

//      sqLoss += sqLossAdd;
//      totalLoss += sqLossAdd;
    }

    double totalLossInc = lambda
        * (frobNormSquare(LTable, LRowBegin, LRowEnd, K) + frobNormSquare(
            RTable, RRowBegin, RRowEnd, K));
    
    lossRecorder.incLoss(ithEval, "FullLoss", totalLossInc);//could be commented

//    lossRecorder.incLoss(ithEval, "SquareLoss", sqLoss);
//    lossRecorder.incLoss(ithEval, "FullLoss", totalLoss);
    lossRecorder.incLoss(ithEval, "NumSamples", elemEnd - elemBegin);
  }

  private static double frobNormSquare(DoubleTable table, int rowBegin,
      int rowEnd, int K) {
    double result = 0;
    for (int i = rowBegin; i < rowEnd; i++) {
      result += frobNormSquare(table.get(i), K);
    }
    return result;
  }

  private static double frobNormSquare(DoubleRow row, int K) {
    double result = 0;
    for (int i = 0; i < K; i++) {
      result += Math.pow(row.get(i), 2);
    }
    return result;
  }

}

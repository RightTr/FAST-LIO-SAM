#pragma once

// Tracks and repairs a sensor timestamp stream when its clock jumps backward.
class TimeSyncronizer
{
public:
    bool repair(double raw_stamp, double expected_dt, double &corrected_stamp,
                bool &rolled_back)
    {
        rolled_back = last_raw_stamp_ >= 0.0 && raw_stamp < last_raw_stamp_;
        corrected_stamp = raw_stamp + stamp_offset_;

        if (last_corrected_stamp_ < 0.0)
        {
            last_corrected_stamp_ = corrected_stamp;
            last_raw_stamp_ = raw_stamp;
            return true;
        }

        if (raw_stamp < last_raw_stamp_)
        {
            stamp_offset_ += last_corrected_stamp_ + expected_dt - corrected_stamp;
            corrected_stamp = raw_stamp + stamp_offset_;
        }
        else if (stamp_offset_ != 0.0 && raw_stamp >= last_corrected_stamp_)
        {
            stamp_offset_ = 0.0;
            corrected_stamp = raw_stamp;
        }

        if (corrected_stamp <= last_corrected_stamp_)
            return false;

        last_corrected_stamp_ = corrected_stamp;
        last_raw_stamp_ = raw_stamp;
        return true;
    }

    void accept(double raw_stamp)
    {
        last_raw_stamp_ = raw_stamp;
        last_corrected_stamp_ = raw_stamp;
    }

    double offset() const { return stamp_offset_; }
    double lastCorrectedStamp() const { return last_corrected_stamp_; }

private:
    double stamp_offset_ = 0.0;
    double last_raw_stamp_ = -1.0;
    double last_corrected_stamp_ = -1.0;
};

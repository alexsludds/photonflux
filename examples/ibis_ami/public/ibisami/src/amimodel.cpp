/*! \file amimodel.cpp
*   \brief Implementation of AMIModel class.
*
* Original author: David Banas<br>
* Original date:   May 2, 2015
*
* Copyright (c) 2015 David Banas; all rights reserved World wide.
*/

#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "include/amimodel.h"

#define EPS_REL 0.0001

std::ostringstream debug_stream;

//! Initialize the model.
void AMIModel::init(double *impulse_matrix, const long number_of_rows,
    const long aggressors, const double sample_interval,
    const double bit_time, const std::string& AMI_parameters_in) {
    debug_stream.str("");
    impulse_matrix_ = impulse_matrix;
    number_of_rows_ = number_of_rows;
    aggressors_ = aggressors;
    sample_interval_ = sample_interval;
    bit_time_ = bit_time;
    samples_per_bit_ = long(bit_time / sample_interval + 0.5);
    msg_ = "Input parameter string: " + AMI_parameters_in + "\n";
    if (std::abs(samples_per_bit_ * sample_interval - bit_time) > (bit_time * EPS_REL))
        msg_ += "WARN: Non-integral number of samples per bit detected!\n";
    gen_data_cnt_ = 0;
    shift_reg_[0] = 1;
    for (auto i = 1; i < PRBS_LEN; i++)
        shift_reg_[i] = 0;
    gen_data_last_ = -1.0;
    param_tree_.name = "";
    param_tree_.children.clear();
    ParseRes res = parse_params(AMI_parameters_in);
    if (!res.first)
        throw std::runtime_error(res.second);
}

//! Parse the incoming AMI parameter string.
ParseRes AMIModel::parse_params(const std::string& AMI_parameters_in) {
    using boost::spirit::ascii::space;
    using boost::spirit::qi::phrase_parse;

    ibisami::AMIGrammar<std::string::const_iterator> the_grammar;
    std::string::const_iterator iter(AMI_parameters_in.begin());
    std::string::const_iterator end(AMI_parameters_in.end());

    bool res = phrase_parse(iter, end, the_grammar, space, param_tree_);
    if (!res || iter != end)
        return ParseRes(false, std::string(iter, end));

    name_ = param_tree_.name;
    return ParseRes(true, std::string(""));
}

/// Returns the integer value of a particular Integer parameter.
/**
* Inputs:
*  - node_names: A vector of strings containing the AMI parameter tree node
*                names required to traverse our way to the parameter of
*                interest. The root name should not be included.
*
*  - default_val: The value to return, if the parameter is not found in the tree.
*
* Returns:
*  - the requested parameter's integer value, if the parameter was found and
*    an integer was able to be scanned from its value string.
*  - 'default_val', if the parameter was not found in the tree.
*
* Throws:
*  - std::runtime_error, if the parameter was found and an integer could not
*    be scanned from its value string.
*/
long AMIModel::get_param_int(const std::vector<std::string>& node_names,
                             long default_val) const {
    std::string param_val_str = get_param(node_names);
    if (param_val_str == "")
        return default_val;
    long res;
    int scans = sscanf(param_val_str.c_str(), "%ld", &res);
    if (scans == 1) {
        return res;
    } else {
        std::ostringstream err;
        err << "AMIModel::get_param_int() could not scan an integer from '"
        << param_val_str << "', while fetching parameter: ";
        std::string err_str = err.str();
        for (std::string node_name : node_names)
            err_str += node_name + ".";
        err_str += "\n";
        throw std::runtime_error(err_str);
    }
}

/// Returns the floating point value of a particular Float parameter.
/**
* Inputs:
*  - node_names: A vector of strings containing the AMI parameter tree node
*                names required to traverse our way to the parameter of
*                interest. The root name should not be included.
*
*  - default_val: The value to return, if the parameter is not found in the tree.
*
* Returns:
*  - the requested parameter's floating point value, if the parameter was found and
*    a double was able to be scanned from its value string.
*  - 'default_val', if the parameter was not found in the tree.
*
* Throws:
*  - std::runtime_error, if the parameter was found and a double could not
*    be scanned from its value string.
*/
double AMIModel::get_param_float(const std::vector<std::string>& node_names,
                               double default_val) const {
    std::string param_val_str = get_param(node_names);
    if (param_val_str == "")
        return default_val;
    double res;
    int scans = sscanf(param_val_str.c_str(), "%lg", &res);
    if (scans == 1) {
        return res;
    } else {
        std::ostringstream err;
        err << "AMIModel::get_param_float() could not scan a double from '"
        << param_val_str << "', while fetching parameter: ";
        std::string err_str = err.str();
        for (std::string node_name : node_names)
            err_str += node_name + ".";
        err_str += "\n";
        throw std::runtime_error(err_str);
    }
}

/// Returns the Boolean value of a particular Bool parameter.
/**
* Inputs:
*  - node_names: A vector of strings containing the AMI parameter tree node
*                names required to traverse our way to the parameter of
*                interest. The root name should not be included.
*
*  - default_val: The value to return, if the parameter is not found in the tree.
*
* Returns:
*  - the requested parameter's Boolean value, if the parameter was found and
*    a Boolean was able to be scanned from its value string.
*  - 'default_val', if the parameter was not found in the tree.
*
* Throws:
*  - std::runtime_error, if the parameter was found and a Boolean could not
*    be scanned from its value string.
*/
bool AMIModel::get_param_bool(const std::vector<std::string>& node_names,
                             bool default_val) const {
    std::string param_val_str = get_param(node_names);
    if (param_val_str == "")
        return default_val;
    if (param_val_str == "True")
        return true;
    else if (param_val_str == "False")
        return false;
    else {
        std::ostringstream err;
        err << "AMIModel::get_param_bool() could not scan a Boolean from '"
        << param_val_str << "', while fetching parameter: ";
        std::string err_str = err.str();
        for (std::string node_name : node_names)
            err_str += node_name + ".";
        err_str += "\n";
        throw std::runtime_error(err_str);
    }
}

/// Returns the string value of a particular AMI parameter.
/**
* Inputs:
*  - node_names: A vector of strings containing the AMI parameter tree node
*                names required to traverse our way to the parameter of
*                interest. The root name should not be included.
*
* Returns:
*  - the requested parameter's value string, if the parameter was found.
*  - empty string, otherwise.
*/
std::string AMIModel::get_param(const std::vector<std::string>& node_names) const {
    return get_leaf(param_tree_, &node_names[0], node_names.size());
}

/// Recursively searches the parameter tree, returning the string value of a particular parameter, if found.
/**
* Inputs:
*  - param_tree: The instance of ibisami::ParamTree to search.
*
*  - node_names: A vector of strings containing the AMI parameter tree node
*                names required to traverse our way to the parameter of
*                interest.
*
* Returns:
*  - The requested parameter's value string, if the parameter was found.
*  - An empty string, otherwise.
*/
std::string AMIModel::get_leaf(const ibisami::ParamTree& param_tree,
                               const std::string* name_ptr,
                               size_t num_names) const {
    const ibisami::ParamTree *next_subtree;
    for (const ibisami::ParamNode& node : param_tree.children) {
        next_subtree = boost::apply_visitor(ibisami::Node(), node);
        if (next_subtree) {
            if (next_subtree->name == *name_ptr) {
                if (num_names == 1)
                    return boost::apply_visitor(ibisami::Value(), next_subtree->children[0]);
                else
                    return get_leaf(*next_subtree, ++name_ptr, num_names - 1);
            }
        }
    }
    return "";
}

/// Generate random binary oversampled data.
/**
 * \param res_vec A pointer to an array of doubles to be overwritten.
 * (It is assumed that, at least, number_of_rows_ elements have been allocated.)
 *
 * The default implementation uses PRBS-7, [7,6].
 */
void AMIModel::gen_data(double *res_vec) {
    unsigned long samples_written = 0;
    while (samples_written++ < number_of_rows_) {
        if (!(gen_data_cnt_++ % samples_per_bit_)) {
            int feedback = (shift_reg_[PRBS_LEN - 1] + shift_reg_[PRBS_LEN - 2]) % 2;
            for (auto i = PRBS_LEN - 1; i > 0; i--)
                shift_reg_[i] = shift_reg_[i - 1];
            shift_reg_[0] = feedback;
            if (shift_reg_[PRBS_LEN - 1])
                gen_data_last_ = 1.0;
            else
                gen_data_last_ = -1.0;
        }
        *res_vec++ = gen_data_last_;
    }
}


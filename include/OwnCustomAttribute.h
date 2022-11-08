#ifndef CUSTOM_ATTRIBUTE_H__
#define CUSTOM_ATTRIBUTE_H__

#include <cassert>
#include <cfloat>
#include <climits>
#include <limits>
#include <cmath>
#include <math.h>
#include <string>
#include <exception>
#include <sstream>
#include <memory>
#include <iostream>
#include <ctime>
#include <fstream>
#include <queue>
#include <functional>
#include <algorithm>
#include <type_traits>
#include <map>

#pragma once

typedef unsigned int uint;
using namespace std;

/** Struct for (maximum) three dimensions indexes. Convention adopted:
 ** i - iterates along the X direction (m_dimX).
 ** j - iterates along the Y direction (m_dimY).
 ** k - iterates along the Z direction (m_dimZ). */

struct dimensions_t {
	int x, y, z;

	dimensions_t() {
		x = y = z = 0;
	}

	dimensions_t(int gX, int gY) {
		x = gX; y = gY; z = 0;
	}

	dimensions_t(int gX, int gY, int gZ) {
		x = gX; y = gY; z = gZ;
	}

	dimensions_t(const dimensions_t& rhs) {
		x = rhs.x; y = rhs.y; z = rhs.z;
	}

	/************************************************************************/
	/* Operators                                                            */
	/************************************************************************/
	// Operators
	// Array indexing
	inline int& operator [] (unsigned int i) {
		assert(i < 3);
		return *(&x + i);
	}

	// Array indexing
	inline const int& operator [] (unsigned int i) const {
		assert(i < 3);
		return *(&x + i);
	}

	dimensions_t operator +(dimensions_t rhs) {
		return dimensions_t(x + rhs.x, y + rhs.y, z + rhs.z);
	}

	inline dimensions_t& operator+=(const dimensions_t& rhs) {
		x += rhs.x; y += rhs.y; z += rhs.z;
		return *this;
	}

	inline dimensions_t friend operator +(const dimensions_t& lhs, const dimensions_t& rhs) {
		dimensions_t dim(lhs);
		dim += rhs;
		return dim;
	}

	inline bool operator==(const dimensions_t& rhs) {
		return(x == rhs.x && y == rhs.y && z == rhs.z);
	}

	inline bool friend operator==(const dimensions_t& lhs, const dimensions_t& rhs) {
		return(lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
	}

	inline bool operator!=(const dimensions_t& rhs) {
		return !(*this == rhs);
	}

	inline bool friend operator!=(const dimensions_t& lhs, const dimensions_t& rhs) {
		return !(lhs == rhs);
	}

	inline bool operator<(const dimensions_t& rhs) {
		if (this->y < rhs.y)
			return true;
		else if (this->y == rhs.y && this->x < rhs.x)
			return true;
		else
			return false;
	}

	inline bool friend operator<(const dimensions_t& lhs, const dimensions_t& rhs) {
		if (lhs.y < rhs.y)
			return true;
		else if (lhs.y == rhs.y && lhs.x < rhs.x)
			return true;
		else
			return false;
	}
	inline bool operator>(const dimensions_t& rhs) {
		if (this->y > rhs.y)
			return true;
		else if (this->y == rhs.y && this->x > rhs.x)
			return true;
		else
			return false;
	}

	inline bool friend operator>(const dimensions_t& lhs, const dimensions_t& rhs) {
		if (lhs.y > rhs.y)
			return true;
		else if (lhs.y == rhs.y && lhs.x > rhs.x)
			return true;
		else
			return false;
	}
};

/** Classes can inherit this class for having custom attribute interfaces and members functionalities added to
	them. Multiple different attribute types are added through multiple inheritance.*/
template <class AttributeClassT>
class OwnCustomAttribute {

	public:
	#pragma region Constructors
	OwnCustomAttribute() { }


	OwnCustomAttribute(const string& attributesName, const AttributeClassT& attribute) {
		setAttribute(attributesName, attribute);
	}

	OwnCustomAttribute(const vector<string>& attributesNames, const vector<AttributeClassT>& attributes) {
		for (uint i = 0; i < attributesNames.size(); i++) {
			setAttribute(attributesNames[i], attributes[i]);
		}
	}

	#pragma endregion

	#pragma region AccessFunctions

	/** Adds an attributes based an empty constructor */
	void addAttributes(const vector<string>& attributeNames) {
		for (auto attributeName : attributeNames) {
			string lowerCase(attributeName);
			transform(lowerCase.begin(), lowerCase.end(), lowerCase.begin(), ::tolower);
			m_attributes[lowerCase] = AttributeClassT();
		}
	}

	/** Returns attribute by name. Does not error treat, in favor of efficiency */
	AttributeClassT& getAttribute(const string& attributeName) {
		string lowerCase(attributeName);
		transform(lowerCase.begin(), lowerCase.end(), lowerCase.begin(), ::tolower);
		return m_attributes[lowerCase];
	}

	/** Returns attribute by name. Does not error treat, in favor of efficiency */
	const AttributeClassT& getAttribute(const string& attributeName) const {
		string lowerCase(attributeName);
		transform(lowerCase.begin(), lowerCase.end(), lowerCase.begin(), ::tolower);
		return m_attributes.at(lowerCase);
	}

	/** Sets attribute by name. It converts strings to lower case, to reduce mismatching.
			Does not error treat, in favor of efficiency*/
	void setAttribute(const string& attributeName, const AttributeClassT& attribute) {
		string lowerCase(attributeName);
		transform(lowerCase.begin(), lowerCase.end(), lowerCase.begin(), ::tolower);
		m_attributes[lowerCase] = attribute;
	}

	virtual bool hasAttribute(const string& attributeName) {
		string lowerCase(attributeName);
		transform(lowerCase.begin(), lowerCase.end(), lowerCase.begin(), ::tolower);
		if (m_attributes.find(lowerCase) == m_attributes.end())
			return false;
		return true;
	}

	/** Clears attributes */
	virtual void clearAttributes() {
		m_attributes.clear();
	}

	const map<string, AttributeClassT>& getAttributesMap() const {
		return m_attributes;
	}

	map<string, AttributeClassT>& getAttributesMap() {
		return m_attributes;
	}

	#pragma endregion

	#pragma region Functionalities
	template <class ParentClassT>
	static OwnCustomAttribute<AttributeClassT>* get(shared_ptr<ParentClassT> pParent) {
		return dynamic_cast<OwnCustomAttribute<AttributeClassT>*>(pParent.get());
	}
	template <class ParentClassT>
	static OwnCustomAttribute<AttributeClassT>* get(ParentClassT* pParent) {
		return dynamic_cast<OwnCustomAttribute<AttributeClassT>*>(pParent);
	}
	#pragma endregion
	
	protected:
	#pragma region ClassMembers
	/* Custom attributes map*/
	map<string, AttributeClassT> m_attributes;
	#pragma endregion

};

template <class AttributeClassT>
using OwnCustomVectorAttribute = OwnCustomAttribute<vector<AttributeClassT>>;

#endif

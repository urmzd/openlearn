// vector.hpp - a std::vector work-alike.
//
// The reference implementation. main.cpp tests it.
//
// The reason this is exercise 01 and not a warm-up: a correct Vector has to
// separate ALLOCATION from CONSTRUCTION. std::vector holds capacity worth of
// raw storage and only constructs objects in the first `size` slots. Get that
// wrong and you default-construct objects nobody asked for, destroy objects
// twice, or leak the ones past the end.
#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

template <typename T>
class Vector {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using iterator = T*;
  using const_iterator = const T*;

  Vector() noexcept = default;

  explicit Vector(size_type n, const T& value = T()) {
    reserve(n);
    for (size_type i = 0; i < n; ++i) push_back(value);
  }

  Vector(std::initializer_list<T> init) {
    reserve(init.size());
    for (const T& v : init) push_back(v);
  }

  // ---- the rule of five -------------------------------------------------

  Vector(const Vector& other) {
    reserve(other.size_);
    for (size_type i = 0; i < other.size_; ++i) {
      // Construct in place from the source element. Assigning would require
      // the destination to already hold a live object, and it does not.
      std::construct_at(data_ + i, other.data_[i]);
      ++size_;  // bump per element, so a throwing copy destroys only what exists
    }
  }

  Vector(Vector&& other) noexcept {
    // Steal the buffer and leave the source empty but valid. Note there is no
    // allocation here at all, which is the entire point of a move.
    data_ = other.data_;
    size_ = other.size_;
    cap_ = other.cap_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.cap_ = 0;
  }

  Vector& operator=(const Vector& other) {
    // Copy-and-swap. Self-assignment falls out for free, and the assignment
    // either fully succeeds or leaves *this untouched, because the copy is
    // built before anything of ours is destroyed.
    if (this != &other) {
      Vector tmp(other);
      swap(tmp);
    }
    return *this;
  }

  Vector& operator=(Vector&& other) noexcept {
    if (this != &other) {
      clear();
      operator delete(static_cast<void*>(data_), std::align_val_t{alignof(T)});
      data_ = other.data_;
      size_ = other.size_;
      cap_ = other.cap_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.cap_ = 0;
    }
    return *this;
  }

  ~Vector() {
    clear();  // destroy the live objects...
    operator delete(static_cast<void*>(data_), std::align_val_t{alignof(T)});
    // ...then release the raw storage. Two steps, because they are two things.
  }

  // ---- capacity ---------------------------------------------------------

  size_type size() const noexcept { return size_; }
  size_type capacity() const noexcept { return cap_; }
  bool empty() const noexcept { return size_ == 0; }

  void reserve(size_type n) {
    if (n <= cap_) return;  // reserve never shrinks

    // Allocate raw storage. operator new, not new[]: new[] would construct n
    // objects, which is exactly what a vector must not do to its spare slots.
    void* raw = operator new(n * sizeof(T), std::align_val_t{alignof(T)});
    T* fresh = static_cast<T*>(raw);

    // Move the existing elements across, but only if moving cannot throw.
    // If T's move constructor can throw, a half-finished move would leave
    // elements destroyed in the old buffer and not yet built in the new one,
    // with no way back. Copying instead keeps the originals intact until the
    // new buffer is complete. This is what move_if_noexcept encodes, and it is
    // why adding noexcept to a move constructor measurably speeds up vectors.
    size_type built = 0;
    try {
      for (; built < size_; ++built) {
        std::construct_at(fresh + built, std::move_if_noexcept(data_[built]));
      }
    } catch (...) {
      for (size_type i = 0; i < built; ++i) std::destroy_at(fresh + i);
      operator delete(raw, std::align_val_t{alignof(T)});
      throw;
    }

    for (size_type i = 0; i < size_; ++i) std::destroy_at(data_ + i);
    operator delete(static_cast<void*>(data_), std::align_val_t{alignof(T)});

    data_ = fresh;
    cap_ = n;
  }

  // ---- modifiers --------------------------------------------------------

  void push_back(const T& value) { emplace_back(value); }
  void push_back(T&& value) { emplace_back(std::move(value)); }

  template <typename... Args>
  T& emplace_back(Args&&... args) {
    if (size_ == cap_) grow();
    std::construct_at(data_ + size_, std::forward<Args>(args)...);
    // Increment only AFTER the constructor returns. If it throws, this element
    // never existed and must not be destroyed by clear() or the destructor.
    ++size_;
    return data_[size_ - 1];
  }

  void pop_back() {
    // Destroy the object but keep the capacity: the slot goes back to being
    // raw storage, not to holding a stale T.
    --size_;
    std::destroy_at(data_ + size_);
  }

  void clear() noexcept {
    for (size_type i = 0; i < size_; ++i) std::destroy_at(data_ + i);
    size_ = 0;
  }

  void swap(Vector& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(cap_, other.cap_);
  }

  // ---- access -----------------------------------------------------------

  T& operator[](size_type i) { return data_[i]; }
  const T& operator[](size_type i) const { return data_[i]; }

  T& at(size_type i) {
    if (i >= size_) throw std::out_of_range("Vector::at");
    return data_[i];
  }
  const T& at(size_type i) const {
    if (i >= size_) throw std::out_of_range("Vector::at");
    return data_[i];
  }

  T& front() { return data_[0]; }
  T& back() { return data_[size_ - 1]; }
  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  iterator begin() noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator end() const noexcept { return data_ + size_; }

 private:
  void grow() {
    // Doubling gives amortised O(1) push_back. Starting at 1 rather than 0
    // avoids a special case, and the first few pushes are cheap anyway.
    reserve(cap_ == 0 ? 1 : cap_ * 2);
  }

  T* data_ = nullptr;
  size_type size_ = 0;
  size_type cap_ = 0;
};

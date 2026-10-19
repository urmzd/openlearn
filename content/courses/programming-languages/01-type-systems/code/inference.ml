(* Type inference in OCaml -- the ML that Hindley-Milner was built for.
 *
 * OCaml infers a principal type for nearly every expression with no
 * annotations: the `let` keyword is the generalization point, exactly as in
 * the Algorithm W implementation in hindley_milner.py. This file shows the
 * SAME ideas from the consumer side (let-polymorphism, parametric types,
 * variants as sum types) plus the axis ML adds on top of plain HM: a *module
 * system* with functors -- parametric polymorphism over whole modules.
 *
 * The (* : type *) comments are what the compiler INFERS; you never write them.
 *
 *   Run:    ocaml inference.ml
 *   Check:  ocamlfind ocamlopt inference.ml  (or: ocaml -stop-after typing) *)

(* --------------------------------------------------------------------------- *)
(* 1. Let-polymorphism: inferred, generalized, reused at many types            *)
(* --------------------------------------------------------------------------- *)

(* identity is inferred as 'a -> 'a; compose as ('a -> 'b) -> ('c -> 'a) -> 'c -> 'b
 * -- precisely the principal types our Algorithm W prints. No annotations. *)
let identity x = x                       (* : 'a -> 'a *)
let compose f g x = f (g x)              (* : ('a -> 'b) -> ('c -> 'a) -> 'c -> 'b *)

(* A let-bound polymorphic value can be USED at two different types in one
 * scope -- the whole point of generalization. *)
let demo_let_poly () =
  let id = identity in
  (id 42, id "hello")                    (* id at int AND at string *)

(* --------------------------------------------------------------------------- *)
(* 2. Parametric data types: a polymorphic binary tree                         *)
(* --------------------------------------------------------------------------- *)

type 'a tree =
  | Leaf
  | Node of 'a tree * 'a * 'a tree

let rec insert x = function          (* : 'a -> 'a tree -> 'a tree *)
  | Leaf -> Node (Leaf, x, Leaf)
  | Node (l, v, r) as t ->
    if x < v then Node (insert x l, v, r)
    else if x > v then Node (l, v, insert x r)
    else t

let rec to_list = function           (* : 'a tree -> 'a list *)
  | Leaf -> []
  | Node (l, v, r) -> to_list l @ [ v ] @ to_list r

(* --------------------------------------------------------------------------- *)
(* 3. Ad-hoc polymorphism via variants + pattern matching (exhaustive)         *)
(* --------------------------------------------------------------------------- *)

type shape =
  | Circle of float
  | Rectangle of float * float
  | Triangle of float * float

(* The match is checked for exhaustiveness: drop a case and OCaml warns. *)
let area = function                  (* : shape -> float *)
  | Circle r -> Float.pi *. r *. r
  | Rectangle (w, h) -> w *. h
  | Triangle (b, h) -> 0.5 *. b *. h

(* --------------------------------------------------------------------------- *)
(* 4. Functors: parametric polymorphism over MODULES                           *)
(* --------------------------------------------------------------------------- *)
(* A functor is a function from modules to modules -- abstraction at a level
 * above values and types. MakeStack takes any module providing a `to_string`
 * and produces a Stack module specialized to that element type. This is ML's
 * answer to "generic library parameterized by an interface", distinct from
 * (and older than) typeclasses/traits. *)

module type SHOWABLE = sig
  type t
  val to_string : t -> string
end

module MakeStack (E : SHOWABLE) = struct
  type t = E.t list
  let empty : t = []
  let push x s = x :: s
  let pop = function [] -> None | x :: rest -> Some (x, rest)
  let show s = "[" ^ String.concat "; " (List.map E.to_string s) ^ "]"
end

module IntStack = MakeStack (struct
  type t = int
  let to_string = string_of_int
end)

(* --------------------------------------------------------------------------- *)
(* Demo                                                                         *)
(* --------------------------------------------------------------------------- *)

let () =
  print_endline "1. Let-polymorphism: id used at int AND string";
  let n, s = demo_let_poly () in
  Printf.printf "   id 42 = %d, id \"hello\" = %s\n" n s;

  print_endline "\n2. Parametric tree: inferred 'a tree, used at int and string";
  let nums = List.fold_right insert [ 5; 3; 8; 1; 4; 7; 9 ] Leaf in
  let strs = List.fold_right insert [ "pear"; "apple"; "fig"; "kiwi" ] Leaf in
  Printf.printf "   inorder(int tree)    = [%s]\n"
    (String.concat "; " (List.map string_of_int (to_list nums)));
  Printf.printf "   inorder(string tree) = [%s]\n"
    (String.concat "; " (to_list strs));

  print_endline "\n3. Ad-hoc: area dispatches over an exhaustive variant match";
  List.iter
    (fun (label, sh) -> Printf.printf "   %-9s area = %.3f\n" label (area sh))
    [ ("circle", Circle 2.); ("rectangle", Rectangle (3., 4.)); ("triangle", Triangle (6., 2.)) ];

  print_endline "\n4. Functor: MakeStack(Int) -> a type-safe IntStack module";
  let st = IntStack.push 3 (IntStack.push 2 (IntStack.push 1 IntStack.empty)) in
  Printf.printf "   IntStack = %s\n" (IntStack.show st);
  (match IntStack.pop st with
   | Some (top, _) -> Printf.printf "   pop -> %d\n" top
   | None -> print_endline "   empty");

  print_endline "\nOK"

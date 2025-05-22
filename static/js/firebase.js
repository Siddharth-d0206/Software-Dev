import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,  
  signInWithEmailAndPassword,
  signOut
} from "https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js";
import {
  getFirestore,
  collection,
  addDoc
} from "https://www.gstatic.com/firebasejs/9.6.1/firebase-firestore.js";

const firebaseConfig = {
//firebase config details.
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

export {
  auth,
  db,
  getAuth, 
  signOut,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  collection,
  addDoc
};
